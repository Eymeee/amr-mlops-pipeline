"""FastAPI inference service for the AMR outcome demo model."""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/best_model.pkl")
TRAINING_SUMMARY_PATH = Path("models/training_summary.json")
PREPROCESSING_METADATA_PATH = Path("data/processed/preprocessing_metadata.json")
METRICS_REGISTRY = CollectorRegistry()
API_REQUESTS_TOTAL = Counter(
    "amr_api_requests_total",
    "Total number of FastAPI requests handled by endpoint.",
    ["endpoint"],
    registry=METRICS_REGISTRY,
)
API_PREDICTION_LATENCY_SECONDS = Histogram(
    "amr_api_prediction_latency_seconds",
    "FastAPI prediction request latency in seconds.",
    registry=METRICS_REGISTRY,
)

CLINICAL_WARNING = (
    "This model is a technical MLOps demo and is not clinically valid. "
    "Dataset signal audit found no meaningful association between available "
    "features and Outcome."
)

ANTIBIOTIC_VALUE = Literal["Sensitive", "Intermediate", "Resistant"]


class ArtifactLoadError(RuntimeError):
    """Raised when the API cannot load required local model artifacts."""


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: Literal["Female", "Male"]
    specimen_type: Literal["Blood", "Sputum", "Stool", "Urine", "Wound swab"]
    amoxicillin: ANTIBIOTIC_VALUE
    ciprofloxacin: ANTIBIOTIC_VALUE
    meropenem: ANTIBIOTIC_VALUE
    vancomycin: ANTIBIOTIC_VALUE
    colistin: ANTIBIOTIC_VALUE
    test_method: Literal["Automated System", "Disc Diffusion", "MIC"]
    resistance_genes: Literal["KPC", "NDM-1", "None", "OXA-48", "VIM"]


class ModelInfo(BaseModel):
    selected_model: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]


class PredictionResponse(BaseModel):
    prediction: str
    prediction_id: int
    probabilities: dict[str, float]
    warning: str
    model_info: ModelInfo


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    metadata_loaded: bool
    selected_model: str
    warning: str


class ModelInfoResponse(BaseModel):
    selected_model: str
    model_registry_name: str | None
    primary_metric: str | None
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_schema: list[str]
    class_labels: dict[str, str]
    warning: str


class ServingArtifacts:
    def __init__(
        self,
        *,
        model: Any,
        preprocessing_metadata: dict[str, Any],
        training_summary: dict[str, Any],
    ) -> None:
        self.model = model
        self.preprocessing_metadata = preprocessing_metadata
        self.training_summary = training_summary
        self.feature_columns = preprocessing_metadata["feature_columns"]
        self.target_mapping = preprocessing_metadata["mappings"]["target"]
        self.inverse_target_mapping = {
            int(encoded): label for label, encoded in self.target_mapping.items()
        }
        self._validate_contract()

    def _validate_contract(self) -> None:
        summary_features = self.training_summary.get("feature_columns")
        if summary_features and summary_features != self.feature_columns:
            raise ArtifactLoadError(
                "Feature mismatch between preprocessing metadata and training summary."
            )

        expected_features = [
            "Age",
            "Gender",
            "Specimen_Type",
            "Amoxicillin",
            "Ciprofloxacin",
            "Meropenem",
            "Vancomycin",
            "Colistin",
            "Test_Method",
            "Resistance_Genes",
        ]
        if self.feature_columns != expected_features:
            raise ArtifactLoadError(
                f"Unsupported feature schema: {self.feature_columns}"
            )

        if set(self.inverse_target_mapping) != {0, 1, 2}:
            raise ArtifactLoadError(
                f"Unsupported target mapping: {self.target_mapping}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactLoadError(f"Missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise ArtifactLoadError(f"Missing required model artifact: {path}")
    with path.open("rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def load_artifacts() -> ServingArtifacts:
    return ServingArtifacts(
        model=_load_pickle(MODEL_PATH),
        preprocessing_metadata=_read_json(PREPROCESSING_METADATA_PATH),
        training_summary=_read_json(TRAINING_SUMMARY_PATH),
    )


def _encode_request(
    request: PredictionRequest,
    artifacts: ServingArtifacts,
) -> pd.DataFrame:
    mappings = artifacts.preprocessing_metadata["mappings"]
    row = {
        "Age": request.age,
        "Gender": mappings["categoricals"]["Gender"][request.gender],
        "Specimen_Type": mappings["categoricals"]["Specimen_Type"][
            request.specimen_type
        ],
        "Amoxicillin": mappings["antibiotics"][request.amoxicillin],
        "Ciprofloxacin": mappings["antibiotics"][request.ciprofloxacin],
        "Meropenem": mappings["antibiotics"][request.meropenem],
        "Vancomycin": mappings["antibiotics"][request.vancomycin],
        "Colistin": mappings["antibiotics"][request.colistin],
        "Test_Method": mappings["categoricals"]["Test_Method"][request.test_method],
        "Resistance_Genes": mappings["categoricals"]["Resistance_Genes"][
            request.resistance_genes
        ],
    }
    return pd.DataFrame([row], columns=artifacts.feature_columns)


def _model_info(artifacts: ServingArtifacts) -> ModelInfo:
    return ModelInfo(
        selected_model=artifacts.training_summary["selected_model"],
        validation_metrics=artifacts.training_summary["validation_metrics"],
        test_metrics=artifacts.training_summary["test_metrics"],
    )


def _probability_mapping(
    probabilities: np.ndarray,
    artifacts: ServingArtifacts,
) -> dict[str, float]:
    return {
        artifacts.inverse_target_mapping[class_id]: float(probabilities[class_id])
        for class_id in sorted(artifacts.inverse_target_mapping)
    }


app = FastAPI(
    title="AMR Outcome Classifier API",
    description=CLINICAL_WARNING,
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    API_REQUESTS_TOTAL.labels(endpoint="/health").inc()
    artifacts = load_artifacts()
    return HealthResponse(
        status="ready",
        model_loaded=True,
        metadata_loaded=True,
        selected_model=artifacts.training_summary["selected_model"],
        warning=CLINICAL_WARNING,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    API_REQUESTS_TOTAL.labels(endpoint="/model-info").inc()
    artifacts = load_artifacts()
    return ModelInfoResponse(
        selected_model=artifacts.training_summary["selected_model"],
        model_registry_name=artifacts.training_summary.get("model_registry_name"),
        primary_metric=artifacts.training_summary.get("primary_metric"),
        validation_metrics=artifacts.training_summary["validation_metrics"],
        test_metrics=artifacts.training_summary["test_metrics"],
        feature_schema=artifacts.feature_columns,
        class_labels={
            str(class_id): label
            for class_id, label in sorted(artifacts.inverse_target_mapping.items())
        },
        warning=CLINICAL_WARNING,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    API_REQUESTS_TOTAL.labels(endpoint="/predict").inc()
    with API_PREDICTION_LATENCY_SECONDS.time():
        artifacts = load_artifacts()
        features = _encode_request(request, artifacts)
        prediction_id = int(np.asarray(artifacts.model.predict(features)).reshape(-1)[0])
        probabilities = np.asarray(artifacts.model.predict_proba(features))[0]

    return PredictionResponse(
        prediction=artifacts.inverse_target_mapping[prediction_id],
        prediction_id=prediction_id,
        probabilities=_probability_mapping(probabilities, artifacts),
        warning=CLINICAL_WARNING,
        model_info=_model_info(artifacts),
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(
        content=generate_latest(METRICS_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
