"""Generate a data drift report and expose demo Prometheus metrics."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


PROCESSED_DIR = Path("data/processed")
REFERENCE_DATA_PATH = PROCESSED_DIR / "train.csv"
CURRENT_DATA_PATH = PROCESSED_DIR / "test.csv"
REPORT_PATH = Path("monitoring/reports/drift_report.html")
TARGET_COLUMN = "Outcome"
METRICS_PORT = 8001
METRICS_REFRESH_SECONDS = 30


def load_monitoring_data(
    reference_path: Path = REFERENCE_DATA_PATH,
    current_path: Path = CURRENT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference and current datasets, excluding the target column."""
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference dataset not found: {reference_path}")
    if not current_path.exists():
        raise FileNotFoundError(f"Current dataset not found: {current_path}")

    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)

    if TARGET_COLUMN in reference_data.columns:
        reference_data = reference_data.drop(columns=[TARGET_COLUMN])
    if TARGET_COLUMN in current_data.columns:
        current_data = current_data.drop(columns=[TARGET_COLUMN])

    return reference_data, current_data


def run_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_path: Path = REPORT_PATH,
) -> tuple[int, float]:
    """Run Evidently data drift report, save HTML, and return drift count/ratio."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    snapshot.save_html(str(report_path))
    logger.info("Saved drift report to %s", report_path)

    drifted_count = extract_drifted_feature_count(snapshot.dict())
    total_features = len(reference_data.columns)
    drifted_ratio = drifted_count / total_features if total_features else 0.0
    logger.info(
        "Detected %s drifted feature(s) out of %s",
        drifted_count,
        total_features,
    )

    return drifted_count, drifted_ratio


def extract_drifted_feature_count(report_result: dict[str, Any]) -> int:
    """Extract the DriftedColumnsCount value from Evidently report results."""
    for metric in report_result.get("metrics", []):
        config = metric.get("config", {})
        if config.get("type") != "evidently:metric_v2:DriftedColumnsCount":
            continue

        value = metric.get("value", {})
        return int(value.get("count", 0))

    raise ValueError("Could not find DriftedColumnsCount metric in Evidently report.")


def create_prometheus_metrics(
    registry: CollectorRegistry,
) -> tuple[Counter, Gauge, Histogram]:
    predictions_total = Counter(
        "amr_predictions_total",
        "Total number of simulated AMR predictions.",
        registry=registry,
    )
    drifted_features_ratio = Gauge(
        "amr_drifted_features_ratio",
        "Ratio of drifted features detected by Evidently.",
        registry=registry,
    )
    prediction_latency_seconds = Histogram(
        "amr_prediction_latency_seconds",
        "Simulated AMR prediction latency in seconds.",
        registry=registry,
    )

    return predictions_total, drifted_features_ratio, prediction_latency_seconds


def populate_metrics(
    *,
    current_data: pd.DataFrame,
    drifted_ratio: float,
    predictions_total: Counter,
    drifted_features_ratio: Gauge,
    prediction_latency_seconds: Histogram,
) -> None:
    drifted_features_ratio.set(drifted_ratio)

    for _ in current_data.itertuples(index=False):
        predictions_total.inc()
        prediction_latency_seconds.observe(random.uniform(0.01, 0.1))


def run_once() -> None:
    reference_data, current_data = load_monitoring_data()
    _, drifted_ratio = run_drift_report(reference_data, current_data)

    registry = CollectorRegistry()
    predictions_total, drifted_features_ratio, prediction_latency_seconds = (
        create_prometheus_metrics(registry)
    )
    populate_metrics(
        current_data=current_data,
        drifted_ratio=drifted_ratio,
        predictions_total=predictions_total,
        drifted_features_ratio=drifted_features_ratio,
        prediction_latency_seconds=prediction_latency_seconds,
    )

    start_http_server(METRICS_PORT, registry=registry)
    logger.info(
        "Prometheus metrics available on http://127.0.0.1:%s/metrics",
        METRICS_PORT,
    )
    while True:
        time.sleep(METRICS_REFRESH_SECONDS)


def main() -> None:
    run_once()


if __name__ == "__main__":
    main()
