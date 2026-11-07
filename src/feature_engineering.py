import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL_COLUMNS = ["ocean_proximity"]
LOG_COLUMNS = ["total_rooms", "total_bedrooms", "population", "households", "median_income"]
RATIO_COLUMNS = [
    "room_per_household",
    "bedroom_per_household",
    "bedroom_totalroom_ratio",
    "population_per_household",
]
MODEL_NUMERIC_COLUMNS = NUMERIC_COLUMNS + [
    f"{column}_log" for column in LOG_COLUMNS
] + RATIO_COLUMNS


class HousingFeatureEngineer(TransformerMixin, BaseEstimator):
    """Create deterministic numeric features before learned preprocessing."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HousingFeatureEngineer":
        del y
        self._validate_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X)
        features = X.copy()

        for column in LOG_COLUMNS:
            features[f"{column}_log"] = np.log1p(features[column])

        features["room_per_household"] = self._safe_ratio(
            features["total_rooms"], features["households"]
        )
        features["bedroom_per_household"] = self._safe_ratio(
            features["total_bedrooms"], features["households"]
        )
        features["bedroom_totalroom_ratio"] = self._safe_ratio(
            features["total_bedrooms"], features["total_rooms"]
        )
        features["population_per_household"] = self._safe_ratio(
            features["population"], features["households"]
        )
        return features

    @staticmethod
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return numerator / denominator.replace(0, np.nan)

    @staticmethod
    def _validate_columns(features: pd.DataFrame) -> None:
        required = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
        missing = sorted(required.difference(features.columns))
        if missing:
            raise ValueError(f"Input data is missing required columns: {missing}")