"""Benchmark gradient boosting models for AMR outcome classification."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import catboost as cb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.preprocess import METADATA_FILENAME, PROCESSED_DIR  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_MODELS_DIR = Path("models")
DEFAULT_N_TRIALS = 30 
DEFAULT_RANDOM_STATE = 42
DEFAULT_EXPERIMENT_NAME = "amr-outcome-classification"
MODEL_REGISTRY_NAME = "amr-outcome-classifier"

CLASS_LABELS = [0, 1, 2]
PRIMARY_METRIC = "f1_macro"


@dataclass(frozen=True)
class DatasetSplits:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list[str]
    target_column: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build_model: Callable[[dict[str, Any], int], Any]
    suggest_params: Callable[[optuna.Trial], dict[str, Any]]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def load_splits(processed_dir: Path) -> DatasetSplits:
    metadata_path = processed_dir / METADATA_FILENAME
    metadata = load_json(metadata_path)
    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]

    split_frames = {}
    for split_name in ["train", "val", "test"]:
        split_path = processed_dir / f"{split_name}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing processed split: {split_path}")
        split_frames[split_name] = pd.read_csv(split_path)

    for split_name, df in split_frames.items():
        missing_columns = set(feature_columns + [target_column]) - set(df.columns)
        if missing_columns:
            raise ValueError(f"{split_name}.csv is missing columns: {missing_columns}")

    train_df = split_frames["train"]
    val_df = split_frames["val"]
    test_df = split_frames["test"]

    return DatasetSplits(
        x_train=train_df[feature_columns],
        y_train=train_df[target_column],
        x_val=val_df[feature_columns],
        y_val=val_df[target_column],
        x_test=test_df[feature_columns],
        y_test=test_df[target_column],
        feature_columns=feature_columns,
        target_column=target_column,
        metadata=metadata,
    )


def suggest_lightgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def build_lightgbm_model(params: dict[str, Any], random_state: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        **params,
        objective="multiclass",
        num_class=len(CLASS_LABELS),
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def suggest_xgboost_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }


def build_xgboost_model(params: dict[str, Any], random_state: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **params,
        objective="multi:softprob",
        num_class=len(CLASS_LABELS),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


def suggest_catboost_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
    }


def build_catboost_model(params: dict[str, Any], random_state: int) -> cb.CatBoostClassifier:
    return cb.CatBoostClassifier(
        **params,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=random_state,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("lightgbm", build_lightgbm_model, suggest_lightgbm_params),
        ModelSpec("xgboost", build_xgboost_model, suggest_xgboost_params),
        ModelSpec("catboost", build_catboost_model, suggest_catboost_params),
    ]


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc_roc_ovr": float(
            roc_auc_score(
                y_true,
                y_proba,
                average="macro",
                multi_class="ovr",
                labels=CLASS_LABELS,
            )
        ),
    }


def evaluate_model(model: Any, x_data: pd.DataFrame, y_data: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(x_data)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_proba = model.predict_proba(x_data)
    metrics = compute_metrics(y_data, y_pred, y_proba)
    matrix = confusion_matrix(y_data, y_pred, labels=CLASS_LABELS)
    return {
        "metrics": metrics,
        "confusion_matrix": matrix,
    }


def write_confusion_matrix(matrix: np.ndarray, output_path: Path) -> Path:
    df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in CLASS_LABELS],
        columns=[f"predicted_{label}" for label in CLASS_LABELS],
    )
    df.to_csv(output_path)
    return output_path


def log_confusion_matrix(matrix: np.ndarray, artifact_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = write_confusion_matrix(matrix, Path(tmp_dir) / artifact_name)
        mlflow.log_artifact(str(artifact_path), artifact_path="confusion_matrices")


def log_metrics(metrics: dict[str, float], prefix: str | None = None) -> None:
    for metric_name, value in metrics.items():
        key = f"{prefix}_{metric_name}" if prefix else metric_name
        mlflow.log_metric(key, value)


def train_and_evaluate(
    spec: ModelSpec,
    params: dict[str, Any],
    splits: DatasetSplits,
    random_state: int,
) -> tuple[Any, dict[str, Any]]:
    model = spec.build_model(params, random_state)
    model.fit(splits.x_train, splits.y_train)
    evaluation = evaluate_model(model, splits.x_val, splits.y_val)
    return model, evaluation


def optimize_model(
    spec: ModelSpec,
    splits: DatasetSplits,
    *,
    n_trials: int,
    random_state: int,
) -> dict[str, Any]:
    logger.info("Starting Optuna study for %s with %s trials", spec.name, n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest_params(trial)
        with mlflow.start_run(run_name=f"{spec.name}-trial-{trial.number}", nested=True):
            mlflow.set_tag("run_type", "trial")
            mlflow.set_tag("model_name", spec.name)
            mlflow.log_param("model_name", spec.name)
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(params)

            _, evaluation = train_and_evaluate(spec, params, splits, random_state)
            metrics = evaluation["metrics"]
            log_metrics(metrics, prefix="val")
            log_confusion_matrix(
                evaluation["confusion_matrix"],
                artifact_name=f"{spec.name}_trial_{trial.number}_val.csv",
            )

            return metrics[PRIMARY_METRIC]

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{spec.name}-study",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    best_model, best_evaluation = train_and_evaluate(
        spec,
        best_params,
        splits,
        random_state,
    )

    mlflow.set_tag("run_type", "model_final")
    mlflow.set_tag("model_name", spec.name)
    mlflow.log_param("model_name", spec.name)
    mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})
    log_metrics(best_evaluation["metrics"], prefix="val")
    log_confusion_matrix(
        best_evaluation["confusion_matrix"],
        artifact_name=f"{spec.name}_best_val.csv",
    )

    return {
        "model_name": spec.name,
        "model": best_model,
        "best_params": best_params,
        "best_trial_number": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "validation_metrics": best_evaluation["metrics"],
        "validation_confusion_matrix": best_evaluation["confusion_matrix"],
        "n_trials": int(n_trials),
    }


def select_best_model(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        model_results,
        key=lambda result: result["validation_metrics"][PRIMARY_METRIC],
    )


def save_pickle(model: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(model, file)
    return output_path


def serializable_model_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": result["model_name"],
        "best_params": result["best_params"],
        "best_trial_number": result["best_trial_number"],
        "best_value": result["best_value"],
        "validation_metrics": result["validation_metrics"],
        "n_trials": result["n_trials"],
    }


def save_training_artifacts(
    *,
    models_dir: Path,
    best_result: dict[str, Any],
    test_evaluation: dict[str, Any],
    all_results: list[dict[str, Any]],
    args: argparse.Namespace,
    splits: DatasetSplits,
) -> dict[str, Any]:
    models_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = save_pickle(best_result["model"], models_dir / "best_model.pkl")
    val_cm_path = write_confusion_matrix(
        best_result["validation_confusion_matrix"],
        models_dir / "confusion_matrix_val.csv",
    )
    test_cm_path = write_confusion_matrix(
        test_evaluation["confusion_matrix"],
        models_dir / "confusion_matrix_test.csv",
    )

    summary = {
        "model_registry_name": MODEL_REGISTRY_NAME,
        "experiment_name": args.experiment_name,
        "primary_metric": PRIMARY_METRIC,
        "selected_model": best_result["model_name"],
        "random_state": int(args.random_state),
        "n_trials": int(args.n_trials),
        "feature_columns": splits.feature_columns,
        "target_column": splits.target_column,
        "validation_metrics": best_result["validation_metrics"],
        "test_metrics": test_evaluation["metrics"],
        "model_results": [serializable_model_result(result) for result in all_results],
        "artifacts": {
            "best_model": str(best_model_path),
            "training_summary": str(models_dir / "training_summary.json"),
            "confusion_matrix_val": str(val_cm_path),
            "confusion_matrix_test": str(test_cm_path),
        },
    }

    summary_path = models_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_benchmark(
    *,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    n_trials: int = DEFAULT_N_TRIALS,
    random_state: int = DEFAULT_RANDOM_STATE,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> dict[str, Any]:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    splits = load_splits(processed_dir)
    mlflow.set_experiment(experiment_name)

    args = argparse.Namespace(
        processed_dir=processed_dir,
        models_dir=models_dir,
        n_trials=n_trials,
        random_state=random_state,
        experiment_name=experiment_name,
    )

    with mlflow.start_run(run_name="gradient-boosting-benchmark") as parent_run:
        mlflow.set_tag("run_type", "benchmark")
        mlflow.log_param("processed_dir", str(processed_dir))
        mlflow.log_param("models_dir", str(models_dir))
        mlflow.log_param("n_trials_per_model", n_trials)
        mlflow.log_param("random_state", random_state)
        mlflow.log_dict(splits.metadata, "preprocessing_metadata.json")

        model_results = []
        for spec in model_specs():
            with mlflow.start_run(run_name=f"{spec.name}-final", nested=True) as model_run:
                result = optimize_model(
                    spec,
                    splits,
                    n_trials=n_trials,
                    random_state=random_state,
                )
                result["mlflow_run_id"] = model_run.info.run_id
                model_results.append(result)
                logger.info(
                    "%s best validation %s: %.4f",
                    spec.name,
                    PRIMARY_METRIC,
                    result["validation_metrics"][PRIMARY_METRIC],
                )

        best_result = select_best_model(model_results)
        test_evaluation = evaluate_model(best_result["model"], splits.x_test, splits.y_test)
        summary = save_training_artifacts(
            models_dir=models_dir,
            best_result=best_result,
            test_evaluation=test_evaluation,
            all_results=model_results,
            args=args,
            splits=splits,
        )

        mlflow.set_tag("selected_model", best_result["model_name"])
        mlflow.log_param("selected_model", best_result["model_name"])
        log_metrics(best_result["validation_metrics"], prefix="best_val")
        log_metrics(test_evaluation["metrics"], prefix="test")
        mlflow.log_artifact(str(models_dir / "training_summary.json"))
        mlflow.log_artifact(str(models_dir / "best_model.pkl"))
        mlflow.log_artifact(str(models_dir / "confusion_matrix_val.csv"), artifact_path="confusion_matrices")
        mlflow.log_artifact(str(models_dir / "confusion_matrix_test.csv"), artifact_path="confusion_matrices")
        mlflow.sklearn.log_model(
            sk_model=best_result["model"],
            artifact_path="registered_best_model",
            registered_model_name=MODEL_REGISTRY_NAME,
            input_example=splits.x_train.head(3),
        )

        summary["mlflow_parent_run_id"] = parent_run.info.run_id
        (models_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Selected %s with validation %s %.4f and test %s %.4f",
            best_result["model_name"],
            PRIMARY_METRIC,
            best_result["validation_metrics"][PRIMARY_METRIC],
            PRIMARY_METRIC,
            test_evaluation["metrics"][PRIMARY_METRIC],
        )
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LightGBM, XGBoost, and CatBoost for AMR outcome classification."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Directory containing train/val/test CSVs. Defaults to {PROCESSED_DIR}.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory for local model artifacts. Defaults to {DEFAULT_MODELS_DIR}.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Optuna trials per model. Defaults to {DEFAULT_N_TRIALS}.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed. Defaults to {DEFAULT_RANDOM_STATE}.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"MLflow experiment name. Defaults to {DEFAULT_EXPERIMENT_NAME}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_benchmark(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        n_trials=args.n_trials,
        random_state=args.random_state,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
