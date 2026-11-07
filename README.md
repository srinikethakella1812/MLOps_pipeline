# California Housing Price Prediction

A reproducible machine-learning project that predicts California home values
with linear regression. It turns a notebook-based experiment into a modular,
configurable training pipeline, with automated tests and a Docker workflow.

The project demonstrates core MLOps practices for the model-training
lifecycle: repeatable data splitting, leakage-safe preprocessing, evaluation,
and saving model artifacts. It is a learning project, not a deployed prediction
service.

## MLOps practices demonstrated

- **Reproducible runs:** data paths, target, test split, and random seed are
  configured in `src/config.yaml`.
- **Modular pipeline:** preprocessing, feature engineering, model construction,
  evaluation, and orchestration are separated into reusable Python modules.
- **Leakage-safe preprocessing:** imputation, scaling, and category encoding are
  fit within a scikit-learn pipeline on training data only.
- **Evaluation and artifacts:** holdout MSE, RMSE, and R² are saved alongside
  the fitted model for later reuse.
- **Containerized execution:** Docker packages the training code and runtime
  dependencies for a consistent run on another machine.
- **Automated tests:** unit tests cover feature engineering and model behavior,
  including previously unseen categories.

## Training workflow

The pipeline reads `raw/housing.csv`, validates the target column, and creates a
reproducible train/holdout split. It then fits and evaluates a scikit-learn
pipeline, saving the split data under `processed/` and the model and metrics
under `artifacts/`.

The fitted pipeline includes engineered log and ratio features, median
imputation and scaling for numeric data, most-frequent imputation and one-hot
encoding for categorical data, and a linear regression estimator.

## Project structure

```text
raw/housing.csv               Input housing dataset
processed/                    Generated train and holdout splits
artifacts/                    Generated model and evaluation metrics
notebook/                     Exploratory analysis
src/config.yaml               Dataset, split, and output configuration
src/data_preprocessing.py     Data loading, validation, and splitting
src/feature_engineering.py    Deterministic feature transformations
src/model_building.py         Preprocessing and regression pipeline
src/model_evaluation.py       Holdout regression metrics
src/train.py                  End-to-end training entry point
tests/                        Pipeline tests
Dockerfile                    Containerized training setup
```

Generated splits and model artifacts are ignored by Git and can be recreated
by running the pipeline.

## Run locally (Windows PowerShell)

Use Python 3.11 or newer. From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train
```

To run the tests:

```powershell
python -m unittest discover -s tests -v
```

To change the input path, target column, split size, or random seed, edit
`src/config.yaml`. Relative paths are resolved from the repository root.

## Run with Docker

Build and run the training container from the repository root:

```powershell
docker build -t housing-price-pipeline .
docker run --rm `
  -v "${PWD}\processed:/app/processed" `
  -v "${PWD}\artifacts:/app/artifacts" `
  housing-price-pipeline
```

The bind mounts leave generated splits, the model, and metrics on the host.
The image includes the dataset and Python dependencies.

## Current scope and next steps

This repository focuses on repeatable model training and evaluation. It does
not currently include a prediction API, CI/CD workflow, model registry, or
production monitoring. Those would be natural next steps for extending the
project toward a production MLOps lifecycle.
