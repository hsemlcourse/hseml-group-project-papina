from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import feature_names, get_model_info, predict


class PredictRequest(BaseModel):
    policy_rate_pct: float = Field(..., description="Current policy rate")
    year: float
    month: float
    population: float
    dataset_version: float
    rate_change_bps: float
    rate_jan2022_base: float
    cumulative_change_since_2022_bps: float
    rate_3m_avg: float
    rate_12m_avg: float
    rate_12m_max: float
    rate_12m_min: float
    is_all_time_high: float
    is_all_time_low: float
    gfc_period: float
    covid_period: float
    hike_cycle_2022_23: float
    fed_rate: float
    spread_vs_fed_bps: float
    row_completeness_pct: float
    policy_rate_pct_lag1: float
    rate_3m_avg_lag1: float
    rate_lag_1: float
    rate_lag_3: float
    rolling_mean_3: float
    rolling_std_3: float
    momentum_3: float


class PredictResponse(BaseModel):
    predicted_class_index: int
    predicted_class: str
    probabilities: Dict[str, float]


app = FastAPI(
    title="Central Bank Rate Change Prediction API",
    description="REST API for inferring the trained model predicting central bank rate change direction.",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> Dict[str, object]:
    return get_model_info()


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    try:
        result = predict(request.dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PredictResponse(
        predicted_class_index=result["predicted_class_index"],
        predicted_class=result["predicted_class"],
        probabilities=result["probabilities"],
    )
