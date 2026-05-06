"""Preprocess the raw AMR tracking CSV for model training.

This step validates the raw dataset, applies deterministic numeric encodings,
and writes stratified train/validation/test CSV files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.ingest import ANTIBIOTIC_COLUMNS, RAW_CSV_PATH, validate_dataset # pyrefly: ignore[missing-import]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


PROCESSED_DIR = Path("data/processed")
METADATA_FILENAME = "preprocessing_metadata.json"
TARGET_COLUMN = "Outcome"
FEATURE_COLUMNS = [
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

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
DEFAULT_RANDOM_STATE = 42

ANTIBIOTIC_MAPPING = {
    "Sensitive": 0,
    "Intermediate": 1,
    "Resistant": 2,
}
CATEGORY_MAPPINGS = {
    "Gender": {
        "Female": 0,
        "Male": 1,
    },
    "Specimen_Type": {
        "Blood": 0,
        "Sputum": 1,
        "Stool": 2,
        "Urine": 3,
        "Wound swab": 4,
    },
    "Test_Method": {
        "Automated System": 0,
        "Disc Diffusion": 1,
        "MIC": 2,
    },
    "Resistance_Genes": {
        "KPC": 0,
        "NDM-1": 1,
        "None": 2,
        "OXA-48": 3,
        "VIM": 4,
    },
}
TARGET_MAPPING = {
    "Recovered": 0,
    "ICU": 1,
    "Deceased": 2,
}


class PreprocessingError(ValueError):
    """Raised when preprocessing cannot safely encode or split the dataset."""


def _read_validated_raw_data(raw_path: Path) -> pd.DataFrame:
    validate_dataset(raw_path)
    return pd.read_csv(raw_path, keep_default_na=False)


def _encode_column(df: pd.DataFrame, column: str, mapping: dict[str, int]) -> None:
    encoded = df[column].map(mapping)
    if encoded.isna().any():
        invalid_values = sorted(df.loc[encoded.isna(), column].astype(str).unique())
        raise PreprocessingError(
            f"Column {column!r} contains values with no encoding: {invalid_values}"
        )

    df[column] = encoded.astype("int64")


def encode_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-model columns and apply stable numeric encodings."""
    encoded_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    for column in ANTIBIOTIC_COLUMNS:
        _encode_column(encoded_df, column, ANTIBIOTIC_MAPPING)

    for column, mapping in CATEGORY_MAPPINGS.items():
        _encode_column(encoded_df, column, mapping)

    _encode_column(encoded_df, TARGET_COLUMN, TARGET_MAPPING)
    encoded_df["Age"] = encoded_df["Age"].astype("int64")

    return encoded_df


def split_dataset(
    df: pd.DataFrame,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic 70/15/15 stratified train/val/test splits."""
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=random_state,
        stratify=temp_df[TARGET_COLUMN],
    )

    return (
        train_df.sort_index().reset_index(drop=True),
        val_df.sort_index().reset_index(drop=True),
        test_df.sort_index().reset_index(drop=True),
    )


def _assert_numeric_columns(df: pd.DataFrame) -> None:
    non_numeric = [
        column
        for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise PreprocessingError(f"Processed data has non-numeric columns: {non_numeric}")


def _write_split(df: pd.DataFrame, output_path: Path) -> None:
    _assert_numeric_columns(df)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %s rows to %s", len(df), output_path)


def _class_distribution(df: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in df[TARGET_COLUMN].value_counts().sort_index().items()
    }


def _build_metadata(
    *,
    raw_path: Path,
    processed_dir: Path,
    random_state: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    output_files = {
        "train": str(processed_dir / "train.csv"),
        "val": str(processed_dir / "val.csv"),
        "test": str(processed_dir / "test.csv"),
        "metadata": str(processed_dir / METADATA_FILENAME),
    }
    split_counts = {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }

    return {
        "raw_path": str(raw_path),
        "processed_dir": str(processed_dir),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "mappings": {
            "antibiotics": ANTIBIOTIC_MAPPING,
            "categoricals": CATEGORY_MAPPINGS,
            "target": TARGET_MAPPING,
        },
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "random_state": int(random_state),
        "split_counts": split_counts,
        "class_distribution": {
            "train": _class_distribution(train_df),
            "val": _class_distribution(val_df),
            "test": _class_distribution(test_df),
        },
        "output_files": output_files,
    }


def preprocess_dataset(
    raw_path: Path = RAW_CSV_PATH,
    processed_dir: Path = PROCESSED_DIR,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Validate, encode, split, and save the AMR dataset."""
    raw_df = _read_validated_raw_data(raw_path)
    encoded_df = encode_dataset(raw_df)
    train_df, val_df, test_df = split_dataset(encoded_df, random_state=random_state)

    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_split(train_df, processed_dir / "train.csv")
    _write_split(val_df, processed_dir / "val.csv")
    _write_split(test_df, processed_dir / "test.csv")

    metadata = _build_metadata(
        raw_path=raw_path,
        processed_dir=processed_dir,
        random_state=random_state,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
    metadata_path = processed_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote preprocessing metadata to %s", metadata_path)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode and split the raw AMR dataset for training."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_CSV_PATH,
        help=f"Path to the raw CSV. Defaults to {RAW_CSV_PATH}.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Directory for processed artifacts. Defaults to {PROCESSED_DIR}.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for deterministic splits. Defaults to {DEFAULT_RANDOM_STATE}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = preprocess_dataset(
        raw_path=args.raw_path,
        processed_dir=args.processed_dir,
        random_state=args.random_state,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
