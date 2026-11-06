from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .control import TrainingControl, _replace_with_retry
from .data import DatasetSplit, TrainingDataset, impute_and_normalize_sequences


class TrainingStopped(RuntimeError):
    """Raised after a safe checkpoint has been written at a stop request."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary, path)


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    selected = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(selected).mean())


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "log_loss": _log_loss(labels, probabilities),
        "accuracy_pct": float((predictions == labels).mean() * 100.0),
        "profitable_outcome_recall_pct": float(
            ((predictions == 2) & (labels == 2)).sum()
            / max(1, (labels == 2).sum())
            * 100.0
        ),
        "profitable_outcome_precision_pct": float(
            ((predictions == 2) & (labels == 2)).sum()
            / max(1, (predictions == 2).sum())
            * 100.0
        ),
    }


def _probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _xgboost_device(xgb: Any) -> str:
    build_info = xgb.build_info() if hasattr(xgb, "build_info") else {}
    return "cuda" if build_info.get("USE_CUDA", False) else "cpu"


def _fixed_class_probabilities(model: Any, values: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(values)
    fixed = np.zeros((len(values), 3), dtype=np.float64)
    for source_index, label in enumerate(model.classes_):
        fixed[:, int(label)] = probabilities[:, source_index]
    return fixed


def train_classical_models(
    dataset: TrainingDataset,
    split: DatasetSplit,
    *,
    run_dir: str | Path,
    control: TrainingControl,
    trees: int = 120,
    patience: int = 30,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Train resumable random/extra tree ensembles and a regularized linear baseline."""
    if trees < 10 or patience < 1:
        raise ValueError("trees must be at least 10 and patience must be positive")
    try:
        import joblib
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Scikit-learn is not installed. Install requirements-training.txt."
        ) from exc

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, np.ndarray]] = {}
    train_values = dataset.features[split.train]
    train_labels = dataset.labels[split.train]
    validation_values = dataset.features[split.validation]
    validation_labels = dataset.labels[split.validation]
    test_values = dataset.features[split.test]
    factories = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=10,
            warm_start=True,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=seed,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=10,
            warm_start=True,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=2,
            random_state=seed,
        ),
    }
    for model_name, factory in factories.items():
        checkpoint_path = output / f"{model_name}.checkpoint.joblib"
        state_path = output / f"{model_name}.state.json"
        final_path = output / f"{model_name}.joblib"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            model = joblib.load(checkpoint_path)
        else:
            model = factory()
            state = {
                "trees_completed": 0,
                "best_loss": math.inf,
                "trees_without_improvement": 0,
                "completed": False,
            }

        def save_tree_checkpoint() -> None:
            temporary_model = checkpoint_path.with_name(
                f"{checkpoint_path.stem}.tmp.joblib"
            )
            joblib.dump(model, temporary_model)
            _replace_with_retry(temporary_model, checkpoint_path)
            _write_json(state_path, state)

        if not state["completed"]:
            tree_count = int(state["trees_completed"])
            while tree_count < trees:
                tree_count = min(tree_count + 10, trees)
                model.set_params(n_estimators=tree_count)
                model.fit(train_values, train_labels)
                probabilities = _fixed_class_probabilities(
                    model,
                    validation_values,
                )
                validation_loss = _log_loss(validation_labels, probabilities)
                state["trees_completed"] = tree_count
                if validation_loss < float(state["best_loss"]):
                    state["best_loss"] = validation_loss
                    state["trees_without_improvement"] = 0
                else:
                    state["trees_without_improvement"] += 10
                save_tree_checkpoint()
                print(
                    f"{model_name}: {tree_count}/{trees} trees, "
                    f"validation log loss {validation_loss:.4f}"
                )
                if not control.checkpoint_boundary(save_tree_checkpoint):
                    raise TrainingStopped(
                        f"{model_name} stopped after saving its checkpoint"
                    )
                if int(state["trees_without_improvement"]) >= patience:
                    break
            state["completed"] = True
            save_tree_checkpoint()
        temporary_final = final_path.with_name(f"{final_path.stem}.tmp.joblib")
        joblib.dump(model, temporary_final)
        _replace_with_retry(temporary_final, final_path)
        validation_probabilities = _fixed_class_probabilities(
            model,
            validation_values,
        )
        test_probabilities = _fixed_class_probabilities(model, test_values)
        _write_json(
            output / f"{model_name}.metrics.json",
            {
                "trees": model.n_estimators,
                "validation": classification_metrics(
                    validation_labels,
                    validation_probabilities,
                ),
                "test": classification_metrics(dataset.labels[split.test], test_probabilities),
            },
        )
        results[model_name] = {
            "validation_probabilities": validation_probabilities,
            "test_probabilities": test_probabilities,
        }

    linear_path = output / "logistic_regression.joblib"
    if linear_path.exists():
        linear_model = joblib.load(linear_path)
    else:
        linear_model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
            ),
        )
        linear_model.fit(train_values, train_labels)
        temporary_linear = linear_path.with_name(
            f"{linear_path.stem}.tmp.joblib"
        )
        joblib.dump(linear_model, temporary_linear)
        _replace_with_retry(temporary_linear, linear_path)
    if not control.checkpoint_boundary(lambda: None):
        raise TrainingStopped("Logistic regression stopped at the model boundary")
    validation_probabilities = _fixed_class_probabilities(
        linear_model[-1],
        linear_model[:-1].transform(validation_values),
    )
    test_probabilities = _fixed_class_probabilities(
        linear_model[-1],
        linear_model[:-1].transform(test_values),
    )
    _write_json(
        output / "logistic_regression.metrics.json",
        {
            "validation": classification_metrics(
                validation_labels,
                validation_probabilities,
            ),
            "test": classification_metrics(
                dataset.labels[split.test],
                test_probabilities,
            ),
        },
    )
    results["logistic_regression"] = {
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
    }
    return results


def train_xgboost(
    dataset: TrainingDataset,
    split: DatasetSplit,
    *,
    run_dir: str | Path,
    control: TrainingControl,
    max_rounds: int = 500,
    patience: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    if max_rounds < 1 or patience < 1:
        raise ValueError("max_rounds and patience must be at least one")
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is not installed. Install the optional training dependencies "
            "from requirements-training.txt."
        ) from exc
    device = _xgboost_device(xgb)

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "xgboost.checkpoint.ubj"
    state_path = output / "xgboost.state.json"
    best_path = output / "xgboost.best.ubj"
    final_path = output / "xgboost.ubj"
    train_matrix = xgb.DMatrix(
        dataset.features[split.train],
        label=dataset.labels[split.train],
        feature_names=list(dataset.feature_names),
    )
    validation_matrix = xgb.DMatrix(
        dataset.features[split.validation],
        label=dataset.labels[split.validation],
        feature_names=list(dataset.feature_names),
    )
    state: dict[str, Any] = {
        "rounds_completed": 0,
        "best_loss": math.inf,
        "best_round": 0,
        "rounds_without_improvement": 0,
        "completed": False,
    }
    booster = None
    if checkpoint_path.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        booster = xgb.Booster()
        booster.load_model(checkpoint_path)
    if state.get("completed") and best_path.exists():
        booster = xgb.Booster()
        booster.load_model(best_path)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": device,
        "max_depth": 4,
        "eta": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "nthread": 2,
    }
    rounds_completed = int(state["rounds_completed"])

    def save_checkpoint() -> None:
        if booster is not None:
            temporary_model = checkpoint_path.with_name(
                f"{checkpoint_path.stem}.tmp.ubj"
            )
            booster.save_model(temporary_model)
            _replace_with_retry(temporary_model, checkpoint_path)
        _write_json(state_path, state)

    if not state.get("completed"):
        for iteration in range(rounds_completed, max_rounds):
            round_params = {**params, "seed": seed + iteration}
            booster = xgb.train(
                round_params,
                train_matrix,
                num_boost_round=1,
                xgb_model=booster,
                verbose_eval=False,
            )
            state["rounds_completed"] = iteration + 1
            probabilities = booster.predict(validation_matrix).reshape(-1, 3)
            validation_loss = _log_loss(
                dataset.labels[split.validation],
                probabilities,
            )
            if validation_loss < float(state["best_loss"]):
                state["best_loss"] = validation_loss
                state["best_round"] = iteration + 1
                state["rounds_without_improvement"] = 0
                temporary_best = best_path.with_name(f"{best_path.stem}.tmp.ubj")
                booster.save_model(temporary_best)
                _replace_with_retry(temporary_best, best_path)
            else:
                state["rounds_without_improvement"] = (
                    int(state["rounds_without_improvement"]) + 1
                )
            save_checkpoint()
            if (iteration + 1) % 25 == 0:
                print(
                    f"XGBoost round {iteration + 1}/{max_rounds}: "
                    f"validation log loss {validation_loss:.4f}"
                )
            if not control.checkpoint_boundary(save_checkpoint):
                raise TrainingStopped("XGBoost stopped after saving its checkpoint")
            if int(state["rounds_without_improvement"]) >= patience:
                break

        state["completed"] = True
        save_checkpoint()
        if not best_path.exists():
            raise RuntimeError("XGBoost completed without a best validation checkpoint")
        shutil.copyfile(best_path, final_path)
    else:
        shutil.copyfile(best_path, final_path)

    best_booster = xgb.Booster()
    best_booster.load_model(final_path)
    validation_probabilities = best_booster.predict(validation_matrix).reshape(-1, 3)
    test_matrix = xgb.DMatrix(
        dataset.features[split.test],
        feature_names=list(dataset.feature_names),
    )
    test_probabilities = best_booster.predict(test_matrix).reshape(-1, 3)
    _write_json(
        output / "xgboost.metrics.json",
        {
            "best_round": state["best_round"],
            "training_device": device,
            "validation": classification_metrics(
                dataset.labels[split.validation],
                validation_probabilities,
            ),
            "test": classification_metrics(
                dataset.labels[split.test],
                test_probabilities,
            ),
        },
    )
    return {
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
    }


def _build_lstm(torch: Any, feature_count: int) -> Any:
    class SmallLSTM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = torch.nn.LSTM(
                input_size=feature_count,
                hidden_size=32,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(32, 5),
            )

        def forward(self, values: Any) -> Any:
            recurrent, _ = self.recurrent(values)
            return self.head(recurrent[:, -1, :])

    return SmallLSTM()


def train_lstm(
    dataset: TrainingDataset,
    split: DatasetSplit,
    *,
    run_dir: str | Path,
    control: TrainingControl,
    epochs: int = 40,
    patience: int = 8,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must be at least one")
    try:
        import psutil
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch or psutil is not installed. Install the optional training "
            "dependencies from requirements-training.txt and select a compatible "
            "Python version."
        ) from exc

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "lstm.checkpoint.pt"
    final_path = output / "lstm.pt"
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and os.name == "nt":
        torch.backends.cudnn.enabled = False
        print(
            "Windows CUDA LSTM training: cuDNN disabled for a clean process exit; "
            "CUDA remains enabled.",
            flush=True,
        )
    print(f"LSTM training device: {device}", flush=True)
    mean = dataset.features[split.train].mean(axis=0).astype(np.float32)
    scale = dataset.features[split.train].std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    normalized_sequences = impute_and_normalize_sequences(
        dataset.sequences,
        mean=mean,
        scale=scale,
    )
    model = _build_lstm(torch, len(dataset.feature_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    return_targets = np.clip(dataset.net_returns_r, -5.0, 5.0)
    process = psutil.Process()
    state: dict[str, Any] = {
        "epoch": 0,
        "batch_offset": 0,
        "best_loss": math.inf,
        "best_epoch": 0,
        "epochs_without_improvement": 0,
        "best_model_state": None,
        "completed": False,
    }

    def save_checkpoint() -> None:
        payload = {
            **state,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        temporary = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
        torch.save(payload, temporary)
        _replace_with_retry(temporary, checkpoint_path)

    def save_final_model() -> None:
        temporary_final = final_path.with_name(f"{final_path.name}.tmp")
        torch.save(
            {
                "model_state": model.state_dict(),
                "feature_names": list(dataset.feature_names),
                "mean": mean,
                "scale": scale,
                "sequence_length": dataset.sequence_length,
                "style": dataset.style,
                "strategy": dataset.strategy,
                "best_epoch": state["best_epoch"],
                "device_used": str(device),
                "return_target_range_r": [-5.0, 5.0],
            },
            temporary_final,
        )
        _replace_with_retry(temporary_final, final_path)

    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        state.update(
            {
                key: checkpoint[key]
                for key in state
                if key in checkpoint
            }
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        np.random.set_state(checkpoint["numpy_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    else:
        save_checkpoint()

    train_indices = split.train
    validation_sequences = torch.from_numpy(
        normalized_sequences[split.validation]
    ).to(device)
    validation_labels = torch.from_numpy(
        dataset.labels[split.validation].astype(np.int64)
    ).to(device)
    validation_returns = torch.from_numpy(
        return_targets[split.validation].astype(np.float32)
    ).to(device)

    def evaluate() -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            outputs = model(validation_sequences)
            log_variance = outputs[:, 4].clamp(-5.0, 3.0)
            return_loss = 0.5 * (
                log_variance
                + (validation_returns - outputs[:, 3]).square()
                * torch.exp(-log_variance)
            ).mean()
            loss = criterion(outputs[:, :3], validation_labels) + 0.1 * return_loss
            probabilities = torch.softmax(outputs[:, :3], dim=1).cpu().numpy()
            expected_returns = outputs[:, 3].clamp(-5.0, 5.0).cpu().numpy()
            return_uncertainty = torch.exp(0.5 * log_variance).cpu().numpy()
        return float(loss.item()), probabilities, expected_returns, return_uncertainty

    def memory_report() -> str:
        memory = psutil.virtual_memory()
        report = (
            f"RAM process {process.memory_info().rss / (1024**3):.2f} GiB, "
            f"system available {memory.available / (1024**3):.2f} GiB"
        )
        if device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            report += (
                f"; GPU allocated {torch.cuda.memory_allocated(device) / (1024**3):.2f}/"
                f"{total_bytes / (1024**3):.2f} GiB, "
                f"free {free_bytes / (1024**3):.2f} GiB"
            )
        return report

    if state["completed"]:
        if final_path.exists():
            saved = torch.load(final_path, map_location=device, weights_only=False)
            model.load_state_dict(saved["model_state"])
        elif state["best_model_state"] is not None:
            model.load_state_dict(state["best_model_state"])
            save_final_model()
        else:
            raise RuntimeError("Completed LSTM checkpoint is missing its best model")
    else:
        for epoch in range(int(state["epoch"]), epochs):
            model.train()
            permutation = np.random.default_rng(seed + epoch).permutation(train_indices)
            start_offset = int(state["batch_offset"]) if epoch == int(state["epoch"]) else 0
            for offset in range(start_offset, len(permutation), batch_size):
                batch_indices = permutation[offset : offset + batch_size]
                batch_x = torch.from_numpy(
                    normalized_sequences[batch_indices]
                ).to(device)
                batch_y = torch.from_numpy(
                    dataset.labels[batch_indices].astype(np.int64)
                ).to(device)
                batch_returns = torch.from_numpy(
                    return_targets[batch_indices].astype(np.float32)
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch_x)
                log_variance = outputs[:, 4].clamp(-5.0, 3.0)
                return_loss = 0.5 * (
                    log_variance
                    + (batch_returns - outputs[:, 3]).square()
                    * torch.exp(-log_variance)
                ).mean()
                loss = criterion(outputs[:, :3], batch_y) + 0.1 * return_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                state["epoch"] = epoch
                state["batch_offset"] = offset + batch_size
                if not control.checkpoint_boundary(save_checkpoint):
                    raise TrainingStopped("LSTM stopped after saving its checkpoint")
                if state["batch_offset"] % (batch_size * 10) == 0:
                    save_checkpoint()

            validation_loss, _, _, _ = evaluate()
            if validation_loss < float(state["best_loss"]):
                state["best_loss"] = validation_loss
                state["best_epoch"] = epoch + 1
                state["epochs_without_improvement"] = 0
                state["best_model_state"] = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                state["epochs_without_improvement"] = (
                    int(state["epochs_without_improvement"]) + 1
                )
            state["epoch"] = epoch + 1
            state["batch_offset"] = 0
            save_checkpoint()
            print(
                f"LSTM epoch {epoch + 1}/{epochs}: "
                f"validation log loss {validation_loss:.4f}; {memory_report()}"
            )
            if int(state["epochs_without_improvement"]) >= patience:
                break

        if state["best_model_state"] is None:
            raise RuntimeError("LSTM completed without a best validation checkpoint")
        model.load_state_dict(state["best_model_state"])
        state["completed"] = True
        save_checkpoint()
        save_final_model()

    (
        _,
        validation_probabilities,
        validation_expected_returns,
        validation_return_uncertainty,
    ) = evaluate()
    test_sequences = torch.from_numpy(
        normalized_sequences[split.test]
    ).to(device)
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_sequences)
        test_probabilities = torch.softmax(test_outputs[:, :3], dim=1).cpu().numpy()
        test_expected_returns = test_outputs[:, 3].clamp(-5.0, 5.0).cpu().numpy()
        test_return_uncertainty = torch.exp(
            0.5 * test_outputs[:, 4].clamp(-5.0, 3.0)
        ).cpu().numpy()

    def return_metrics(
        indices: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> dict[str, float]:
        targets = return_targets[indices]
        errors = predictions - targets
        return {
            "expected_return_mae_r": float(np.abs(errors).mean()),
            "expected_return_rmse_r": float(np.sqrt(np.square(errors).mean())),
            "mean_return_uncertainty_r": float(uncertainties.mean()),
        }

    _write_json(
        output / "lstm.metrics.json",
        {
            "best_epoch": state["best_epoch"],
            "device_used": str(device),
            "validation": classification_metrics(
                dataset.labels[split.validation],
                validation_probabilities,
            ),
            "test": classification_metrics(
                dataset.labels[split.test],
                test_probabilities,
            ),
            "validation_return": return_metrics(
                split.validation,
                validation_expected_returns,
                validation_return_uncertainty,
            ),
            "test_return": return_metrics(
                split.test,
                test_expected_returns,
                test_return_uncertainty,
            ),
        },
    )
    return {
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
        "validation_expected_returns_r": validation_expected_returns,
        "test_expected_returns_r": test_expected_returns,
        "validation_return_uncertainty_r": validation_return_uncertainty,
        "test_return_uncertainty_r": test_return_uncertainty,
        "mean": mean,
        "scale": scale,
    }


def calibrate_hybrid(
    *,
    labels: np.ndarray,
    model_probabilities: dict[str, np.ndarray],
    run_dir: str | Path,
    regime_thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Select model weights and temperature using validation data only."""
    if not model_probabilities:
        raise ValueError("At least one model's validation probabilities are required")
    model_names = tuple(model_probabilities)
    if any(
        values.shape != model_probabilities[model_names[0]].shape
        for values in model_probabilities.values()
    ):
        raise ValueError("Model validation probability shapes do not match")
    if model_probabilities[model_names[0]].shape != (len(labels), 3):
        raise ValueError("Validation probabilities must have three columns per row")

    weight_candidates = [
        *np.eye(len(model_names), dtype=np.float64),
        np.full(len(model_names), 1.0 / len(model_names)),
    ]
    if len(model_names) > 1:
        weight_candidates.extend(
            np.random.default_rng(42).dirichlet(
                np.ones(len(model_names)),
                size=512,
            )
        )
    best: tuple[float, np.ndarray, float] | None = None
    probability_stack = np.stack(
        [model_probabilities[name] for name in model_names]
    )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for hybrid calibration. Install "
            "requirements-training.txt."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hybrid calibration device: {device}", flush=True)
    if device.type == "cuda":
        available_bytes, _ = torch.cuda.mem_get_info(device)
        bytes_per_weight = (
            len(labels)
            * 21
            * 3
            * np.dtype(np.float64).itemsize
            * 3
        )
        chunk_size = max(1, min(8, available_bytes // max(1, bytes_per_weight)))
    else:
        chunk_size = 32

    probability_tensor = torch.as_tensor(
        probability_stack,
        dtype=torch.float64,
        device=device,
    )
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
    temperatures = torch.as_tensor(
        np.linspace(0.6, 1.6, 21),
        dtype=torch.float64,
        device=device,
    )
    weight_array = np.asarray(weight_candidates)
    with torch.no_grad():
        for start in range(0, len(weight_candidates), chunk_size):
            candidate_weights = torch.as_tensor(
                weight_array[start : start + chunk_size],
                dtype=torch.float64,
                device=device,
            )
            blended = torch.einsum(
                "bm,mnc->bnc",
                candidate_weights,
                probability_tensor,
            )
            logits = torch.log(blended.clamp(min=1e-12, max=1.0))
            scaled_logits = logits[:, None, :, :] / temperatures[
                None, :, None, None
            ]
            calibrated = torch.softmax(scaled_logits, dim=-1)
            target_probabilities = calibrated.gather(
                dim=-1,
                index=labels_tensor[None, None, :, None].expand(
                    len(candidate_weights),
                    len(temperatures),
                    -1,
                    1,
                ),
            ).squeeze(-1)
            losses = -torch.log(target_probabilities.clamp_min(1e-12)).mean(dim=-1)
            flat_index = int(losses.argmin().item())
            batch_index, temperature_index = divmod(flat_index, len(temperatures))
            loss = float(losses[batch_index, temperature_index].item())
            if best is None or loss < best[0]:
                candidate_index = start + batch_index
                best = (
                    loss,
                    weight_array[candidate_index].copy(),
                    float(temperatures[temperature_index].item()),
                )
    if best is None:
        raise RuntimeError("Hybrid calibration did not produce a validation score")

    weights = best[1]
    blended = np.tensordot(weights, probability_stack, axes=(0, 0))
    calibrated = _probabilities_from_logits(
        np.log(np.clip(blended, 1e-12, 1.0)),
        best[2],
    )
    result = {
        "format_version": 1,
        "calibration_device": str(device),
        "model_weights": {
            name: float(weight) for name, weight in zip(model_names, weights)
        },
        "temperature": best[2],
        "validation_log_loss": best[0],
        "regime_thresholds": regime_thresholds,
        "validation_metrics": classification_metrics(labels, calibrated),
    }
    _write_json(Path(run_dir) / "hybrid.json", result)
    return result
