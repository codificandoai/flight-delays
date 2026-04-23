import unittest
import os
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    FEATURES_COLS = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    TARGET_COL = [
        "delay"
    ]


    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        self.data = pd.read_csv(filepath_or_buffer="../data/data.csv")
        

    def test_model_preprocess_for_training(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)


    def test_model_preprocess_for_serving(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)


    def test_model_fit(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        self.model.fit(
            features=features,
            target=target
        )

        predicted_target = self.model._model.predict(
            features_validation
        )

        report = classification_report(target_validation, predicted_target, output_dict=True)
        
        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30


    def test_model_predict(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        predicted_targets = self.model.predict(
            features=features
        )

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)

    def test_preprocess_api_input(self):
        """Test preprocessing of API-style input (no target column, no Fecha columns)."""
        api_data = pd.DataFrame([
            {"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 12},
            {"OPERA": "Latin American Wings", "TIPOVUELO": "N", "MES": 7},
            {"OPERA": "Sky Airline", "TIPOVUELO": "I", "MES": 4},
        ])
        features = self.model.preprocess(api_data)
        assert isinstance(features, pd.DataFrame)
        assert features.shape == (3, len(self.FEATURES_COLS))
        assert set(features.columns) == set(self.FEATURES_COLS)

    def test_feature_alignment(self):
        """Test that unknown OPERA/MES values produce all-zero rows aligned to TOP_10_FEATURES."""
        api_data = pd.DataFrame([
            {"OPERA": "Unknown Airline", "TIPOVUELO": "N", "MES": 2},
        ])
        features = self.model.preprocess(api_data)
        assert features.shape == (1, len(self.FEATURES_COLS))
        assert (features.iloc[0] == 0).all()

    def test_predict_single_flight(self):
        """Test predict on a single flight returns a list with one int."""
        api_data = pd.DataFrame([
            {"OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 3},
        ])
        features = self.model.preprocess(api_data)
        predictions = self.model.predict(features)
        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert predictions[0] in (0, 1)

    def test_predict_output_values_are_binary(self):
        """Test that all predictions are 0 or 1."""
        features = self.model.preprocess(self.data)
        predictions = self.model.predict(features)
        assert all(p in (0, 1) for p in predictions)

    def test_get_min_diff(self):
        """Test the static _get_min_diff method."""
        row = pd.Series({
            "Fecha-O": "2017-01-01 23:45:00",
            "Fecha-I": "2017-01-01 23:30:00",
        })
        result = DelayModel._get_min_diff(row)
        assert result == 15.0

    def test_preprocess_target_delay_values(self):
        """Test that target column contains only 0 and 1."""
        features, target = self.model.preprocess(self.data, target_column="delay")
        unique_vals = set(target["delay"].unique())
        assert unique_vals.issubset({0, 1})

    def test_model_persistence(self):
        """Test that fit saves the model and predict can load it."""
        features, target = self.model.preprocess(self.data, target_column="delay")
        self.model.fit(features, target)
        assert os.path.exists(DelayModel._MODEL_PATH)

        new_model = DelayModel()
        assert new_model._model is None
        api_data = pd.DataFrame([
            {"OPERA": "Copa Air", "TIPOVUELO": "I", "MES": 10},
        ])
        feats = new_model.preprocess(api_data)
        preds = new_model.predict(feats)
        assert isinstance(preds, list)
        assert len(preds) == 1