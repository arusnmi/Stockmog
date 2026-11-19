@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
if /I "%~1"=="predict-series" (
    python -m data_pipeline.nifty_prediction_series %*
) else if /I "%~1"=="forecast-report" (
    python -m data_pipeline.nifty_forecast_report %*
) else (
    python -m modeling.nifty_timeframe_workflow %*
)
set "result=%errorlevel%"
popd
exit /b %result%
