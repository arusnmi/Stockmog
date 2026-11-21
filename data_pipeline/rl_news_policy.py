"""Maskable PPO policy with small, bounded news and research action-score adjustments."""

from __future__ import annotations

import math

from gymnasium.spaces import Discrete
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import torch
from torch import nn


class NewsPriorityExtractor(nn.Module):
    """Primary actor/critic never see side inputs; news then research follow the actor latent."""

    def __init__(self, features_dim: int, news_indices: list[int],
                 net_arch: dict[str, list[int]], activation_fn: type[nn.Module],
                 research_indices: list[int] | None = None):
        super().__init__()
        research_indices = research_indices or []
        side = news_indices + research_indices
        if (not news_indices or len(set(side)) != len(side)
                or min(side) < 0 or max(side) >= features_dim):
            raise ValueError("News/research indices must be unique, disjoint and within the observation")
        excluded = set(side)
        primary_indices = [index for index in range(features_dim) if index not in excluded]
        if not primary_indices:
            raise ValueError("The policy needs non-news inputs")
        self.register_buffer("primary_indices", torch.tensor(primary_indices, dtype=torch.long))
        self.register_buffer("news_indices", torch.tensor(news_indices, dtype=torch.long))
        self.register_buffer("research_indices", torch.tensor(research_indices, dtype=torch.long))

        def network(widths: list[int]) -> nn.Sequential:
            if not widths or min(widths) < 1:
                raise ValueError("Actor and critic must have positive hidden layers")
            layers: list[nn.Module] = []
            previous = len(primary_indices)
            for width in widths:
                layers.extend([nn.Linear(previous, width), activation_fn()])
                previous = width
            return nn.Sequential(*layers)

        self.policy_net = network(net_arch["pi"])
        self.value_net = network(net_arch["vf"])
        self.primary_actor_dim = net_arch["pi"][-1]
        self.news_dim, self.research_dim = len(news_indices), len(research_indices)
        self.latent_dim_pi = self.primary_actor_dim + self.news_dim + self.research_dim
        self.latent_dim_vf = net_arch["vf"][-1]

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        primary = self.policy_net(features.index_select(1, self.primary_indices))
        return torch.cat([primary, features.index_select(1, self.news_indices),
                          features.index_select(1, self.research_indices)], dim=1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features.index_select(1, self.primary_indices))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class BoundedNewsActionNet(nn.Module):
    def __init__(self, primary_dim: int, news_dim: int, actions: int, limit: float,
                 research_dim: int = 0, research_limit: float = 0.0):
        super().__init__()
        self.primary_dim, self.news_dim, self.limit = primary_dim, news_dim, limit
        self.research_limit = research_limit
        self.primary = nn.Linear(primary_dim, actions)
        self.news = nn.Linear(news_dim, actions, bias=False)
        nn.init.zeros_(self.news.weight)
        self.research = nn.Linear(research_dim, actions, bias=False) if research_dim else None
        if self.research is not None:
            nn.init.zeros_(self.research.weight)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        split = self.primary_dim + self.news_dim
        primary = self.primary(latent[:, :self.primary_dim])
        adjustment = self.limit * torch.tanh(self.news(latent[:, self.primary_dim:split]))
        if self.research is not None:
            adjustment = adjustment + self.research_limit * torch.tanh(self.research(latent[:, split:]))
        return primary + adjustment


class LowPriorityNewsPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, news_indices: list[int], news_logit_limit: float = 0.05,
                 research_indices: list[int] | None = None, research_logit_limit: float = 0.15,
                 **kwargs):
        if not math.isfinite(news_logit_limit) or not 0 <= news_logit_limit <= 0.1:
            raise ValueError("News logit limit must be between zero and 0.1")
        if not math.isfinite(research_logit_limit) or not 0 <= research_logit_limit <= 0.3:
            raise ValueError("Research logit limit must be between zero and 0.3")
        self.news_indices = news_indices
        self.news_logit_limit = news_logit_limit
        self.research_indices = research_indices or []
        self.research_logit_limit = research_logit_limit
        kwargs.setdefault("ortho_init", False)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        if not self.share_features_extractor or not isinstance(self.net_arch, dict):
            raise ValueError("Low-priority news requires shared features and explicit actor/critic layers")
        self.mlp_extractor = NewsPriorityExtractor(
            self.features_dim, self.news_indices, self.net_arch, self.activation_fn,
            self.research_indices)

    def _build(self, lr_schedule) -> None:
        if not isinstance(self.action_space, Discrete):
            raise ValueError("Low-priority news supports discrete buy/sell/stay actions only")
        if self.ortho_init:
            raise ValueError("Low-priority news uses default initialization, not orthogonal initialization")
        self._build_mlp_extractor()
        self.action_net = BoundedNewsActionNet(
            self.mlp_extractor.primary_actor_dim, len(self.news_indices),
            self.action_space.n, self.news_logit_limit,
            len(self.research_indices), self.research_logit_limit)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
