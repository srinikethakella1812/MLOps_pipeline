import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from src.data_preprocessing import load_data, split_data
from src.model_building import build_model
from src.model_evaluation import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    return config


def train(config_path: Path = DEFAULT_CONFIG) -> dict[str, float]:
    """Train, evaluate, and save the configured house-price model."""
    config = _load_config(config_path)
    data_config = config["data"]
    output_config = config["output"]

    target_column = data_config["target_column"]
    test_size = float(data_config["test_size"])
    if not 0 < test_size < 1:
        raise ValueError("data.test_size must be greater than 0 and less than 1")
    random_state = int(data_config["random_state"])

    data_path = _resolve_path(data_config["input"], PROJECT_ROOT)
    data = load_data(data_path, target_column)
    X_train, X_test, y_train, y_test = split_data(
        data,
        target_column,
        test_size,
        random_state,
    )

    processed_config = config["processed_data"]
    train_path = _resolve_path(processed_config["train"], PROJECT_ROOT)
    test_path = _resolve_path(processed_config["test"], PROJECT_ROOT)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train.rename(target_column)], axis=1).to_csv(
        train_path, index=False
    )
    pd.concat([X_test, y_test.rename(target_column)], axis=1).to_csv(
        test_path, index=False
    )

    model = build_model()
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    model_path = _resolve_path(output_config["model"], PROJECT_ROOT)
    metrics_path = _resolve_path(output_config["metrics"], PROJECT_ROOT)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Trained model saved to: {model_path}")
    print(f"Holdout metrics saved to: {metrics_path}")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the house-price model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
