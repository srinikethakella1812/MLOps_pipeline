import unittest

import numpy as np
import pandas as pd

from src.feature_engineering import HousingFeatureEngineer
from src.model_building import build_model


def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [-122.2, -121.8, -119.5, -118.2, -117.1, -122.4],
            "latitude": [37.8, 37.2, 36.8, 34.1, 33.9, 38.0],
            "housing_median_age": [10, 20, 30, 40, 50, 15],
            "total_rooms": [100, 200, 300, 400, 0, 150],
            "total_bedrooms": [20, None, 60, 80, 10, 30],
            "population": [50, 100, 150, 200, 25, 75],
            "households": [10, 20, 0, 40, 5, 15],
            "median_income": [2.0, 3.0, 4.0, 5.0, 1.5, 2.5],
            "ocean_proximity": [
                "NEAR BAY",
                "INLAND",
                "ISLAND",
                "NEAR OCEAN",
                "INLAND",
                "NEAR BAY",
            ],
        }
    )


class HousingPipelineTests(unittest.TestCase):
    def test_feature_engineering_handles_zero_denominators(self) -> None:
        features = sample_features()
        transformed = HousingFeatureEngineer().fit_transform(features)

        self.assertTrue(np.isnan(transformed.loc[2, "room_per_household"]))
        self.assertTrue(np.isnan(transformed.loc[4, "bedroom_totalroom_ratio"]))
        self.assertEqual(len(transformed), len(features))

    def test_pipeline_fits_and_predicts_unseen_category(self) -> None:
        features = sample_features()
        target = pd.Series([100000, 150000, 200000, 250000, 90000, 125000])
        model = build_model()
        model.fit(features.iloc[:5], target.iloc[:5])

        unseen_category = features.iloc[[5]].copy()
        unseen_category.loc[:, "ocean_proximity"] = "NEW CATEGORY"
        predictions = model.predict(unseen_category)

        self.assertEqual(len(predictions), 1)
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
