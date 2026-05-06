"""Download and validate the raw AMR tracking dataset.

The current project dataset is a single CSV from Mendeley Data:
https://data.mendeley.com/datasets/h4byb28gcv/2

This step keeps the raw file untouched and validates that its schema and
category values match the assumptions used by preprocessing and training.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


MENDELEY_DATASET_ID = "h4byb28gcv"
MENDELEY_DATASET_VERSION = 2
MENDELEY_API_URL = (
    f"https://api.mendeley.com/datasets/{MENDELEY_DATASET_ID}"
    f"?version={MENDELEY_DATASET_VERSION}&fields=*"
)

RAW_DIR = Path("data/raw")
RAW_FILENAME = "antibiotic_resistance_tracking.csv"
RAW_CSV_PATH = RAW_DIR / RAW_FILENAME

EXPECTED_ROW_COUNT = 2_200
EXPECTED_COLUMNS = [
    "Patient_ID",
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
    "Outcome",
]

ANTIBIOTIC_COLUMNS = [
    "Amoxicillin",
    "Ciprofloxacin",
    "Meropenem",
    "Vancomycin",
    "Colistin",
]
MODEL_COLUMNS = [column for column in EXPECTED_COLUMNS if column != "Patient_ID"]

ALLOWED_VALUES = {
    "Gender": {"Female", "Male"},
    "Specimen_Type": {"Blood", "Urine", "Sputum", "Wound swab", "Stool"},
    "Amoxicillin": {"Sensitive", "Intermediate", "Resistant"},
    "Ciprofloxacin": {"Sensitive", "Intermediate", "Resistant"},
    "Meropenem": {"Sensitive", "Intermediate", "Resistant"},
    "Vancomycin": {"Sensitive", "Intermediate", "Resistant"},
    "Colistin": {"Sensitive", "Intermediate", "Resistant"},
    "Test_Method": {"Automated System", "MIC", "Disc Diffusion"},
    "Resistance_Genes": {"KPC", "NDM-1", "OXA-48", "VIM", "None"},
    "Outcome": {"Recovered", "ICU", "Deceased"},
}


class DatasetValidationError(ValueError):
    """Raised when the raw dataset does not match the expected contract."""


def _read_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV without treating the category string 'None' as NA."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {csv_path}")

    return pd.read_csv(csv_path, keep_default_na=False)


def _format_values(values: set[Any]) -> str:
    return ", ".join(repr(value) for value in sorted(values))


def _validate_columns(df: pd.DataFrame) -> None:
    actual_columns = list(df.columns)
    if actual_columns == EXPECTED_COLUMNS:
        return

    missing = set(EXPECTED_COLUMNS) - set(actual_columns)
    extra = set(actual_columns) - set(EXPECTED_COLUMNS)
    raise DatasetValidationError(
        "Unexpected dataset columns. "
        f"Expected {EXPECTED_COLUMNS}, got {actual_columns}. "
        f"Missing: {_format_values(missing) or 'none'}. "
        f"Extra: {_format_values(extra) or 'none'}."
    )


def _validate_no_missing_or_blank(df: pd.DataFrame) -> None:
    model_df = df[MODEL_COLUMNS]

    missing_count = int(model_df.isna().sum().sum())
    if missing_count:
        raise DatasetValidationError(
            f"Model columns contain {missing_count} missing values."
        )

    blank_counts = {
        column: int(model_df[column].astype(str).str.strip().eq("").sum())
        for column in model_df.columns
    }
    blank_counts = {column: count for column, count in blank_counts.items() if count}
    if blank_counts:
        raise DatasetValidationError(f"Model columns contain blank values: {blank_counts}")


def _validate_patient_ids(df: pd.DataFrame) -> None:
    patient_ids = df["Patient_ID"].astype(str)
    blank_mask = patient_ids.str.strip().eq("")
    blank_count = int(blank_mask.sum())
    if blank_count:
        logger.warning(
            "Patient_ID contains %s blank value(s). This ID column is not used for "
            "modeling, so ingestion will continue.",
            blank_count,
        )

    patient_ids = patient_ids[~blank_mask]

    duplicate_count = int(patient_ids.duplicated().sum())
    if duplicate_count:
        logger.warning(
            "Patient_ID contains %s duplicate value(s). This ID column is not used for "
            "modeling, so ingestion will continue.",
            duplicate_count,
        )

    invalid_ids = patient_ids[~patient_ids.str.fullmatch(r"P\d{4}")].head(10).tolist()
    if invalid_ids:
        logger.warning(
            "Patient_ID contains values outside the P0001-style pattern. Examples: %s",
            invalid_ids,
        )


def _validate_age(df: pd.DataFrame) -> None:
    if not pd.api.types.is_integer_dtype(df["Age"]):
        raise DatasetValidationError("Age must be an integer column.")

    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())
    if min_age < 1 or max_age > 90:
        raise DatasetValidationError(
            f"Age values must be in the documented 1-90 range. Got {min_age}-{max_age}."
        )


def _validate_allowed_values(df: pd.DataFrame) -> None:
    invalid_values: dict[str, list[str]] = {}
    for column, allowed in ALLOWED_VALUES.items():
        observed = set(df[column].astype(str).str.strip())
        invalid = observed - allowed
        if invalid:
            invalid_values[column] = sorted(invalid)

    if invalid_values:
        raise DatasetValidationError(
            f"Dataset contains unexpected category values: {invalid_values}"
        )


def validate_dataset(csv_path: Path = RAW_CSV_PATH) -> dict[str, Any]:
    """Validate the raw AMR CSV and return a compact ingestion summary."""
    df = _read_raw_csv(csv_path)

    _validate_columns(df)

    if len(df) != EXPECTED_ROW_COUNT:
        raise DatasetValidationError(
            f"Expected {EXPECTED_ROW_COUNT} rows, found {len(df)} rows."
        )

    _validate_no_missing_or_blank(df)
    _validate_patient_ids(df)
    _validate_age(df)
    _validate_allowed_values(df)

    summary = {
        "path": str(csv_path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "age_range": [int(df["Age"].min()), int(df["Age"].max())],
        "blank_patient_ids": int(df["Patient_ID"].astype(str).str.strip().eq("").sum()),
        "duplicate_patient_ids": int(
            df.loc[df["Patient_ID"].astype(str).str.strip().ne(""), "Patient_ID"]
            .astype(str)
            .duplicated()
            .sum()
        ),
        "outcome_distribution": df["Outcome"].value_counts().sort_index().to_dict(),
        "resistance_gene_distribution": (
            df["Resistance_Genes"].value_counts().sort_index().to_dict()
        ),
    }

    logger.info("Validated raw dataset: %s rows, %s columns", df.shape[0], df.shape[1])
    logger.info("Outcome distribution: %s", summary["outcome_distribution"])
    return summary


def _download_file(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading raw dataset from %s", url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        with output_path.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)

    logger.info("Saved raw dataset to %s", output_path)
    return output_path


def _resolve_mendeley_download_url(filename: str = RAW_FILENAME) -> str:
    """Resolve the time-limited Mendeley download URL for the CSV file."""
    logger.info("Resolving Mendeley file metadata from %s", MENDELEY_API_URL)
    response = httpx.get(
        MENDELEY_API_URL,
        headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
        follow_redirects=True,
        timeout=60.0,
    )
    response.raise_for_status()
    dataset = response.json()

    for file_metadata in dataset.get("files", []):
        if file_metadata.get("filename") != filename:
            continue

        content_details = file_metadata.get("content_details", {})
        download_url = content_details.get("download_url")
        if download_url:
            return str(download_url)

    available_files = [
        file_metadata.get("filename", "<unknown>")
        for file_metadata in dataset.get("files", [])
    ]
    raise DatasetValidationError(
        f"Could not find {filename!r} in Mendeley dataset files: {available_files}"
    )


def download_dataset(
    output_path: Path = RAW_CSV_PATH,
    source_url: str | None = None,
) -> Path:
    """Download the raw CSV from a direct URL or from the Mendeley dataset API."""
    download_url = source_url or _resolve_mendeley_download_url()
    return _download_file(download_url, output_path)


def ingest_dataset(
    csv_path: Path = RAW_CSV_PATH,
    *,
    source_url: str | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
    """Ensure the raw CSV exists, then validate it."""
    if force_download or not csv_path.exists():
        download_dataset(output_path=csv_path, source_url=source_url)
    else:
        logger.info("Raw dataset already exists at %s; skipping download", csv_path)

    return validate_dataset(csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest and validate the raw AMR CSV.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=RAW_CSV_PATH,
        help=f"Path to the raw CSV. Defaults to {RAW_CSV_PATH}.",
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help=(
            "Optional direct CSV download URL. If omitted and download is needed, "
            "the script resolves the URL from Mendeley metadata."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the CSV even if it already exists locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = ingest_dataset(
        csv_path=args.csv_path,
        source_url=args.source_url,
        force_download=args.force_download,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
