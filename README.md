# 🧬 AMR MLOps Pipeline

> **Predicting Clinical Outcomes from Antibiotic Resistance Profiles**
> A complete end-to-end MLOps pipeline for clinical decision support in infectious disease management.

---

## 📋 Table of Contents

- [🧬 AMR MLOps Pipeline](#-amr-mlops-pipeline)
  - [📋 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [Clinical Context](#clinical-context)
  - [Dataset](#dataset)
    - [Columns](#columns)
  - [MLOps Pipeline](#mlops-pipeline)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Environment Setup](#environment-setup)
    - [Run the Pipeline](#run-the-pipeline)
  - [ML Model](#ml-model)
    - [Target Classes](#target-classes)
    - [Evaluation Metrics](#evaluation-metrics)
  - [API](#api)
    - [Example Request](#example-request)
    - [Example Response](#example-response)
  - [Monitoring](#monitoring)
  - [CI/CD](#cicd)
  - [Tech Stack](#tech-stack)
  - [Author](#author)

---

## Overview

This project builds a **production-grade MLOps pipeline** that predicts the clinical outcome of hospitalized patients (`Recovered` / `ICU` / `Deceased`) based on their antibiotic resistance profile and demographic characteristics.

The pipeline covers the full ML lifecycle: data ingestion, preprocessing, experiment tracking, model serving, CI/CD automation, and production monitoring.

---

## Clinical Context

Antimicrobial Resistance (AMR) is one of the greatest public health threats of the 21st century — responsible for an estimated 5 million deaths in 2019. When a patient is hospitalized with a bacterial infection, clinicians must quickly determine which antibiotic will be effective. This model supports that decision by predicting patient outcomes based on resistance profiles.

**Target application:** Clinical microbiology departments and therapeutic decision support systems.

---

## Dataset

**Source:** [Antibiotic Resistance Tracking Dataset — Mendeley Data](https://data.mendeley.com/datasets/h4byb28gcv/2)
**License:** CC BY 4.0

| Property | Value |
|---|---|
| Records | 2,200 patients |
| Columns | 12 |
| Missing values | None |
| Format | Single CSV file |

### Columns

| Column | Type | Values |
|---|---|---|
| `Patient_ID` | ID | P0001 → P2200 |
| `Age` | Numeric | 1–90 |
| `Gender` | Categorical | Male / Female |
| `Specimen_Type` | Categorical | Blood, Urine, Sputum, Wound swab, Stool |
| `Amoxicillin` | Categorical | Sensitive / Intermediate / Resistant |
| `Ciprofloxacin` | Categorical | Sensitive / Intermediate / Resistant |
| `Meropenem` | Categorical | Sensitive / Intermediate / Resistant |
| `Vancomycin` | Categorical | Sensitive / Intermediate / Resistant |
| `Colistin` | Categorical | Sensitive / Intermediate / Resistant |
| `Test_Method` | Categorical | Automated System / MIC / Disc Diffusion |
| `Resistance_Genes` | Categorical | KPC, NDM-1, OXA-48, VIM, None |
| `Outcome` ⭐ | **Target** | **Recovered / ICU / Deceased** |

> See [`notebooks/dataset_concepts_en.md`](notebooks/dataset_concepts_en.md) for a detailed explanation of each column.

---

## MLOps Pipeline

```
Mendeley CSV
     │
     ▼
┌─────────────┐
│  1. Ingest  │  Download CSV · DVC versioning
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  2. Preprocess   │  Encoding · Normalization · Train/Val/Test split
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│  3. Version  │  DVC pipelines · Git metadata
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  4. Train    │  LightGBM · Optuna hyperparameter tuning
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  5. Track    │  MLflow experiments · Model Registry
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  6. Serve    │  FastAPI · Docker
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  7. CI/CD    │  GitHub Actions · pytest · Docker Hub
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  8. Monitor  │  Evidently AI · Prometheus · Grafana
└──────────────┘
```

---

## Project Structure

```
amr-mlops-pipeline/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD pipelines
├── context/                # Dataset description and concepts
│   ├── dataset_description.md
│   └── dataset_concepts.md
├── data/
│   ├── raw/                # Raw data from Mendeley (DVC-tracked)
│   └── processed/          # Preprocessed data (DVC-tracked)
├── src/
│   ├── ingestion/          # Data download and inspection
│   │   └── ingest.py
│   ├── preprocessing/      # Feature engineering and encoding
│   │   └── preprocess.py
│   ├── training/           # LightGBM training + Optuna tuning
│   │   └── train.py
│   ├── serving/            # FastAPI REST API
│   │   └── api.py
│   └── monitoring/         # Evidently AI drift detection
│       └── monitor.py
├── tests/                  # pytest unit tests
├── .dvc/                   # DVC configuration
├── .gitignore
├── pyproject.toml          # uv dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Git + DVC
- Docker (for serving)

### Installation

```bash
# Clone the repository
git clone git@github.com:<your-username>/amr-mlops-pipeline.git
cd amr-mlops-pipeline

# Create virtual environment and install dependencies
uv venv --python 3.12
uv add lightgbm scikit-learn pandas numpy optuna \
       fastapi uvicorn mlflow dvc \
       pytest httpx python-dotenv
```

### Environment Setup

```bash
cp .env.example .env
# Fill in your credentials in .env
```

### Run the Pipeline

```bash
# 1. Ingest data
uv run python src/ingestion/ingest.py

# 2. Preprocess
uv run python src/preprocessing/preprocess.py

# 3. Train
uv run python src/training/train.py

# 4. Serve (local)
uv run uvicorn src.serving.api:app --reload

# 5. Run tests
uv run pytest tests/
```

---

## ML Model

| Property | Value |
|---|---|
| Algorithm | LightGBM |
| Task | Multi-class classification (3 classes) |
| Tuning | Optuna (hyperparameter optimization) |
| Tracking | MLflow |
| Hardware | 100% CPU — no GPU required |

### Target Classes

| Class | Description |
|---|---|
| `Recovered` | Patient healed and discharged |
| `ICU` | Patient transferred to intensive care |
| `Deceased` | Patient died from the infection |

### Evaluation Metrics

| Metric | Target |
|---|---|
| F1-Score Macro | > 0.85 |
| AUC-ROC (OvR) | > 0.90 |
| Accuracy | > 80% |

---

## API

Once deployed, the FastAPI server exposes:

```
POST /predict     →  Returns Outcome class + probabilities
GET  /health      →  Health check
GET  /docs        →  Interactive Swagger UI
```

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "gender": "Male",
    "specimen_type": "Blood",
    "amoxicillin": "Resistant",
    "ciprofloxacin": "Resistant",
    "meropenem": "Intermediate",
    "vancomycin": "Sensitive",
    "colistin": "Sensitive",
    "test_method": "Automated System",
    "resistance_genes": "KPC"
  }'
```

### Example Response

```json
{
  "outcome": "ICU",
  "probabilities": {
    "Recovered": 0.21,
    "ICU": 0.61,
    "Deceased": 0.18
  }
}
```

---

## Monitoring

The monitoring stack tracks model behavior in production:

- **Evidently AI** — data drift detection on resistance profiles
- **Prometheus** — metrics collection
- **Grafana** — real-time dashboard with alerts

---

## CI/CD

GitHub Actions pipeline triggered on every push to `main`:

```
push to main
     │
     ├── pytest (unit tests)
     ├── flake8 (linting)
     ├── docker build
     ├── docker push → Docker Hub
     └── deploy to production
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **ML** | LightGBM, Scikit-learn, Optuna |
| **Data Versioning** | DVC, Git |
| **Experiment Tracking** | MLflow |
| **Serving** | FastAPI, Docker, Docker Compose |
| **CI/CD** | GitHub Actions, pytest, flake8 |
| **Monitoring** | Evidently AI, Prometheus, Grafana |
| **Environment** | Python 3.12, uv |

---

## Author

**SALHI Aymane** — MLOps Project 2025-2026
