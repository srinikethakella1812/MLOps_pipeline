from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path: Path, target_column: str) -> pd.DataFrame:
    """Load the CSV and validate that the target is present and complete."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} is missing from {data_path}")
    if data[target_column].isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values")
    return data


def split_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separate features from the target and create a reproducible holdout set."""
    features = data.drop(columns=[target_column])
    target = data[target_column]
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )