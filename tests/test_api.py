from fastapi.testclient import TestClient

from src.serving.api import CLINICAL_WARNING, app


client = TestClient(app)


VALID_PAYLOAD = {
    "age": 36,
    "gender": "Female",
    "specimen_type": "Blood",
    "amoxicillin": "Intermediate",
    "ciprofloxacin": "Sensitive",
    "meropenem": "Intermediate",
    "vancomycin": "Intermediate",
    "colistin": "Intermediate",
    "test_method": "Automated System",
    "resistance_genes": "KPC",
}


def test_health_returns_ready_status_and_warning() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["model_loaded"] is True
    assert payload["metadata_loaded"] is True
    assert payload["warning"] == CLINICAL_WARNING


def test_model_info_exposes_model_metrics_and_warning() -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model"]
    assert "f1_macro" in payload["validation_metrics"]
    assert "f1_macro" in payload["test_metrics"]
    assert payload["class_labels"] == {
        "0": "Recovered",
        "1": "ICU",
        "2": "Deceased",
    }
    assert payload["warning"] == CLINICAL_WARNING


def test_predict_returns_prediction_probabilities_and_warning() -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in {"Recovered", "ICU", "Deceased"}
    assert payload["prediction_id"] in {0, 1, 2}
    assert set(payload["probabilities"]) == {"Recovered", "ICU", "Deceased"}
    assert sum(payload["probabilities"].values()) == 1.0
    assert payload["warning"] == CLINICAL_WARNING
    assert payload["model_info"]["selected_model"]


def test_predict_rejects_invalid_categorical_value() -> None:
    invalid_payload = VALID_PAYLOAD | {"meropenem": "Unknown"}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
