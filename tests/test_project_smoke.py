from src.preprocessing.preprocess import FEATURE_COLUMNS, TARGET_MAPPING # pyrefly: ignore
from src.serving.api import CLINICAL_WARNING # pyrefly: ignore


def test_preprocessing_contract_exports_expected_target_mapping() -> None:
    assert TARGET_MAPPING == {
        "Recovered": 0,
        "ICU": 1,
        "Deceased": 2,
    }


def test_serving_warning_documents_demo_model_limitations() -> None:
    assert FEATURE_COLUMNS
    assert "technical MLOps demo" in CLINICAL_WARNING
    assert "not clinically valid" in CLINICAL_WARNING
