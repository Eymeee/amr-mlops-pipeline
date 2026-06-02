# AMR MLOps Pipeline

End-to-end MLOps demo for antimicrobial resistance data: ingestion, validation,
preprocessing, model benchmarking, experiment tracking, FastAPI serving, Docker,
CI/CD, and monitoring with Prometheus/Grafana.

## Table of Contents

- [AMR MLOps Pipeline](#amr-mlops-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Project Scope](#project-scope)
  - [Dataset](#dataset)
  - [Pipeline Overview](#pipeline-overview)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
  - [Command Shortcuts](#command-shortcuts)
  - [Data Versioning With DVC](#data-versioning-with-dvc)
  - [Run The Pipeline](#run-the-pipeline)
  - [Training And Experiment Tracking](#training-and-experiment-tracking)
  - [Serving API](#serving-api)
  - [Docker Stack](#docker-stack)
  - [Monitoring](#monitoring)
  - [Tests And CI](#tests-and-ci)
  - [Current Status](#current-status)
  - [Tech Stack](#tech-stack)

## Project Scope

The pipeline covers the main lifecycle pieces of a local MLOps project:

- Validate a raw AMR CSV dataset.
- Encode features with deterministic mappings.
- Split data into train, validation, and test sets.
- Track raw and processed data with DVC.
- Benchmark three gradient boosting model families.
- Log experiments, metrics, and artifacts with MLflow.
- Serve the selected model through FastAPI.
- Expose API and monitoring metrics for Prometheus.
- Run a Docker Compose stack for API, monitor, Prometheus, and Grafana.
- Run minimal CI with dependency installation, Ruff, pytest, and Docker build.

The serving layer keeps a warning in every user-facing API response explaining
that predictions are not clinically valid.

## Dataset

| Property | Value |
|---|---|
| Source | Mendeley Data |
| License | CC BY 4.0 |
| Raw file | `data/raw/antibiotic_resistance_tracking.csv` |
| Rows | 2,200 |
| Columns | 12 |
| Target | `Outcome` |
| DVC status | Local-only DVC tracking, no remote configured |

Raw columns:

| Column | Role | Values |
|---|---|---|
| `Patient_ID` | Metadata only | Blank/duplicate values exist, not used for modeling |
| `Age` | Feature | Numeric |
| `Gender` | Feature | `Female`, `Male` |
| `Specimen_Type` | Feature | `Blood`, `Sputum`, `Stool`, `Urine`, `Wound swab` |
| `Amoxicillin` | Feature | `Sensitive`, `Intermediate`, `Resistant` |
| `Ciprofloxacin` | Feature | `Sensitive`, `Intermediate`, `Resistant` |
| `Meropenem` | Feature | `Sensitive`, `Intermediate`, `Resistant` |
| `Vancomycin` | Feature | `Sensitive`, `Intermediate`, `Resistant` |
| `Colistin` | Feature | `Sensitive`, `Intermediate`, `Resistant` |
| `Test_Method` | Feature | `Automated System`, `Disc Diffusion`, `MIC` |
| `Resistance_Genes` | Feature | `KPC`, `NDM-1`, `None`, `OXA-48`, `VIM` |
| `Outcome` | Target | `Recovered`, `ICU`, `Deceased` |

The ingestion step reads the CSV with `keep_default_na=False` so the literal
`Resistance_Genes=None` category is preserved instead of being interpreted as a
missing value.

## Pipeline Overview

```text
Raw CSV
  |
  v
Ingestion validation
  - schema checks
  - row count checks
  - allowed-value checks
  - Patient_ID warnings
  |
  v
Preprocessing
  - drop Patient_ID
  - deterministic categorical encoding
  - stratified train/val/test split
  - metadata JSON
  |
  v
DVC
  - raw CSV tracked by .dvc file
  - preprocessing stage in dvc.yaml
  |
  v
Training
  - LightGBM
  - XGBoost
  - CatBoost
  - Optuna tuning
  - MLflow logging
  |
  v
Serving
  - FastAPI
  - local pickle model artifact
  - preprocessing metadata
  - Prometheus API metrics
  |
  v
Monitoring
  - Evidently drift report
  - Prometheus scraping
  - Grafana visualization
```

## Project Structure

```text
.
|-- .github/workflows/ci.yml
|-- context/
|   `-- modeling_findings.md
|-- data/
|   |-- raw/
|   |   |-- antibiotic_resistance_tracking.csv
|   |   `-- antibiotic_resistance_tracking.csv.dvc
|   `-- processed/
|       |-- train.csv
|       |-- val.csv
|       |-- test.csv
|       `-- preprocessing_metadata.json
|-- monitoring/
|   |-- prometheus.yml
|   `-- reports/
|       `-- drift_report.html
|-- src/
|   |-- ingestion/ingest.py
|   |-- preprocessing/preprocess.py
|   |-- training/train.py
|   |-- serving/api.py
|   `-- monitoring/monitor.py
|-- tests/
|-- Dockerfile
|-- compose.yaml
|-- dvc.yaml
|-- Makefile
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

Some local outputs are intentionally ignored by Git, including `models/`,
`mlruns/`, `mlflow.db`, data artifacts, and generated monitoring reports.

## Setup

Prerequisites:

- Python 3.12
- `uv`
- Git
- DVC
- Docker and Docker Compose for the containerized stack

Install dependencies from the lock file:

```bash
uv sync --frozen
```

If you need to create the environment from scratch:

```bash
uv venv --python 3.12
uv sync --frozen
```

## Command Shortcuts

The root [Makefile](Makefile) wraps the most common local commands.

Show available targets:

```bash
make help
```

Common shortcuts:

```bash
make ingest       # Validate raw data
make preprocess   # Build processed train/val/test splits
make dvc-repro    # Reproduce the DVC preprocessing stage
make train-smoke  # Run a 1-trial training smoke test
make check        # Run Ruff + CI-safe tests
make api          # Start the local FastAPI service
make stack-up     # Start Docker Compose stack
make stack-down   # Stop Docker Compose stack
```

The Docker Compose wrapper defaults to `sudo docker compose` because this local
environment has required sudo. Override it when Docker does not need sudo:

```bash
make stack-up DOCKER_COMPOSE="docker compose"
```

## Data Versioning With DVC

The raw CSV is tracked with DVC:

```bash
data/raw/antibiotic_resistance_tracking.csv.dvc
```

The preprocessing stage is defined in [dvc.yaml](dvc.yaml):

```bash
uv run dvc repro
```

Useful DVC commands:

```bash
uv run dvc status
uv run dvc repro
uv run dvc dag
```

No DVC remote is configured yet. Data is local-only for now.

## Run The Pipeline

Validate the raw dataset:

```bash
uv run python src/ingestion/ingest.py
# or
make ingest
```

Generate processed train/validation/test splits:

```bash
uv run python src/preprocessing/preprocess.py
# or
make preprocess
```

Run the training benchmark:

```bash
uv run python src/training/train.py
# or
make train
```

Run a smaller smoke benchmark:

```bash
uv run python src/training/train.py --n-trials 1
# or
make train-smoke
```

Launch the MLflow UI:

```bash
uv run mlflow ui
# or
make mlflow
```

Then open:

```text
http://127.0.0.1:5000
```

## Training And Experiment Tracking

Training is implemented in [src/training/train.py](src/training/train.py).

The script benchmarks:

- `lightgbm.LGBMClassifier`
- `xgboost.XGBClassifier`
- `catboost.CatBoostClassifier`

Each model gets its own Optuna study. The selected model is chosen by validation
`f1_macro`, then evaluated once on the held-out test set.

Tracked metrics:

- `f1_macro`
- `accuracy`
- `auc_roc_ovr`
- confusion matrix artifacts

Local training outputs:

```text
models/best_model.pkl
models/training_summary.json
models/confusion_matrix_val.csv
models/confusion_matrix_test.csv
```

The best 100-trial run remained near random:

| Metric | Validation | Test |
|---|---:|---:|
| F1 macro | 0.8668 | 0.8139 |
| Accuracy | 0.8727 | 0.8152 |
| AUC ROC OvR | 0.8095 | 0.8994 |


## Serving API

The FastAPI app is implemented in [src/serving/api.py](src/serving/api.py).

Required local artifacts:

```text
models/best_model.pkl
models/training_summary.json
data/processed/preprocessing_metadata.json
```

Run locally:

```bash
uv run uvicorn src.serving.api:app --reload
# or
make api
```

API endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Artifact readiness and selected model |
| `GET` | `/model-info` | Training summary, metrics, schema, class labels |
| `POST` | `/predict` | Single-patient demo prediction |
| `GET` | `/metrics` | Prometheus metrics for the API |
| `GET` | `/docs` | Swagger UI |

Example prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 36,
    "gender": "Female",
    "specimen_type": "Blood",
    "amoxicillin": "Intermediate",
    "ciprofloxacin": "Sensitive",
    "meropenem": "Intermediate",
    "vancomycin": "Intermediate",
    "colistin": "Intermediate",
    "test_method": "Automated System",
    "resistance_genes": "KPC"
  }'
```

Example response shape:

```json
{
  "prediction": "ICU",
  "prediction_id": 1,
  "probabilities": {
    "Recovered": 0.31,
    "ICU": 0.36,
    "Deceased": 0.33
  },
  "warning": "This model is a technical MLOps demo and is not clinically valid. Dataset signal audit found no meaningful association between available features and Outcome.",
  "model_info": {
    "selected_model": "lightgbm",
    "validation_metrics": {
      "f1_macro": 0.3668,
      "accuracy": 0.3727,
      "auc_roc_ovr": 0.5095
    },
    "test_metrics": {
      "f1_macro": 0.3139,
      "accuracy": 0.3152,
      "auc_roc_ovr": 0.4994
    }
  }
}
```

The exact probabilities depend on the local trained model artifact.

## Docker Stack

The Docker image contains source code and dependencies only. It does not bake in
ignored model or data artifacts. Those are mounted at runtime.

Build and run the full stack:

```bash
sudo docker compose up --build
# or
make stack-up
```

Stop the stack:

```bash
sudo docker compose down
# or
make stack-down
```

The current Compose file uses host networking because this local environment had
bridge-network limitations during Docker builds and service scraping.

Runtime mounts:

- `./models:/app/models:ro`
- `./data/processed:/app/data/processed:ro`
- `./monitoring/reports:/app/monitoring/reports`

Useful URLs:

| Service | URL |
|---|---|
| API health | `http://127.0.0.1:8000/health` |
| API docs | `http://127.0.0.1:8000/docs` |
| API metrics | `http://127.0.0.1:8000/metrics` |
| Monitor metrics | `http://127.0.0.1:8001/metrics` |
| Prometheus | `http://127.0.0.1:9090` |
| Prometheus targets | `http://127.0.0.1:9090/targets` |
| Grafana | `http://127.0.0.1:3000` |

Grafana login:

```text
username: admin
password: admin
```

## Monitoring

Monitoring is implemented in [src/monitoring/monitor.py](src/monitoring/monitor.py).

The monitoring script:

- Loads `data/processed/train.csv` as reference data.
- Loads `data/processed/test.csv` as current data.
- Drops the `Outcome` target before drift analysis.
- Runs an Evidently `DataDriftPreset` report.
- Saves the HTML report to `monitoring/reports/drift_report.html`.
- Exposes Prometheus metrics on port `8001`.

Monitoring metrics:

| Metric | Type | Purpose |
|---|---|---|
| `amr_predictions_total` | Counter | Simulated prediction volume |
| `amr_drifted_features_ratio` | Gauge | Ratio of drifted features from Evidently |
| `amr_prediction_latency_seconds` | Histogram | Simulated prediction latency |

The API also exposes:

| Metric | Type | Purpose |
|---|---|---|
| `amr_api_requests_total` | Counter | Requests by API endpoint |
| `amr_api_prediction_latency_seconds` | Histogram | Prediction latency |

Run the monitor directly:

```bash
uv run python src/monitoring/monitor.py
# or
make monitor
```

Prometheus scrape configuration is in
[monitoring/prometheus.yml](monitoring/prometheus.yml).

## Tests And CI

Run all local tests:

```bash
uv run pytest tests/ -v
# or
make test
```

Run CI-equivalent checks locally:

```bash
uv run ruff check src/ tests/
uv run pytest tests/ --ignore=tests/test_api.py -v
docker build -t amr-outcome-api .
```

The Makefile shortcut for the first two CI-equivalent checks is:

```bash
make check
```

`tests/test_api.py` is ignored in CI because it requires local model artifacts
under `models/`, which are not committed to Git.

The GitHub Actions workflow in [.github/workflows/ci.yml](.github/workflows/ci.yml)
runs on push and pull request to `main`:

1. Check out code.
2. Set up Python 3.12.
3. Install `uv`.
4. Install dependencies with `uv sync --frozen`.
5. Run Ruff.
6. Run pytest smoke tests.
7. Build the Docker image.

The CI workflow does not push to Docker Hub and does not deploy.

## Current Status

- Ingestion validation is implemented.
- Preprocessing and deterministic metadata export are implemented.
- DVC tracks the raw CSV and preprocessing stage locally.
- Training benchmarks LightGBM, XGBoost, and CatBoost with Optuna and MLflow.
- Modeling findings are documented and treated as a project constraint.
- FastAPI serving is implemented with clinical-validity warnings.
- API smoke tests pass locally.
- Docker Compose stack is verified locally.
- Makefile command shortcuts are available for local workflows.
- GitHub Actions CI passes.
- Prometheus and Grafana monitoring are verified locally.

## Tech Stack

| Area | Tools |
|---|---|
| Environment | Python 3.12, uv |
| Data | pandas, DVC |
| ML | scikit-learn, LightGBM, XGBoost, CatBoost |
| Optimization | Optuna |
| Tracking | MLflow |
| Serving | FastAPI, Uvicorn |
| Containers | Docker, Docker Compose |
| Monitoring | Evidently, prometheus-client, Prometheus, Grafana |
| Testing and CI | pytest, Ruff, GitHub Actions |
