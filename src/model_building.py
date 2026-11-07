from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import (
    CATEGORICAL_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    HousingFeatureEngineer,
)


def build_model() -> Pipeline:
    """Build a training pipeline that learns preprocessing from training data only."""
    numeric_preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessing, MODEL_NUMERIC_COLUMNS),
            ("categorical", categorical_preprocessing, CATEGORICAL_COLUMNS),
        ]
    )

    return Pipeline(
        steps=[
            ("feature_engineering", HousingFeatureEngineer()),
            ("preprocessing", preprocessing),
            ("regression", LinearRegression()),
        ]
    )