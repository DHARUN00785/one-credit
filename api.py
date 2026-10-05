from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import tensorflow as tf
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from prediction import AGE_OPTIONS, AgeRange, classify_risk, predict_readmission


MODEL_PATH = Path(__file__).resolve().parent / "readmission_model (1).keras"


class PredictionRequest(BaseModel):
    gender: Literal["Female", "Male"]
    age_range: AgeRange
    time_in_hospital: Annotated[int, Field(ge=1, le=14)]
    num_lab_procedures: Annotated[int, Field(ge=1, le=130)]
    num_procedures: Annotated[int, Field(ge=0, le=10)]
    num_medications: Annotated[int, Field(ge=1, le=80)]
    number_outpatient: Annotated[int, Field(ge=0, le=42)]
    number_emergency: Annotated[int, Field(ge=0, le=76)]
    number_inpatient: Annotated[int, Field(ge=0, le=21)]
    number_diagnoses: Annotated[int, Field(ge=1, le=16)]


class PredictionResponse(BaseModel):
    probability: float
    probability_percent: float
    risk_tier: str
    advice: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = tf.keras.models.load_model(MODEL_PATH)
    yield


app = FastAPI(
    title="Patient Readmission Predictor API",
    description="Predict readmission risk from patient demographic and utilization features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    features = [
        1 if payload.gender == "Male" else 0,
        AGE_OPTIONS.index(payload.age_range.value),
        payload.time_in_hospital,
        payload.num_lab_procedures,
        payload.num_procedures,
        payload.num_medications,
        payload.number_outpatient,
        payload.number_emergency,
        payload.number_inpatient,
        payload.number_diagnoses,
    ]
    probability = predict_readmission(request.app.state.model, features)
    risk_tier, _, advice = classify_risk(probability)

    return PredictionResponse(
        probability=probability,
        probability_percent=probability * 100,
        risk_tier=risk_tier,
        advice=advice,
    )