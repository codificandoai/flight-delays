import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb

from datetime import datetime
from typing import Tuple, Union, List


class DelayModel:

    TOP_10_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    _MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model.pkl"
    )

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        features = features.reindex(columns=self.TOP_10_FEATURES, fill_value=0)

        if target_column:
            data = data.copy()
            data["min_diff"] = data.apply(self._get_min_diff, axis=1)
            data[target_column] = np.where(data["min_diff"] > 15, 1, 0)
            target = data[[target_column]]
            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        target_values = target.iloc[:, 0]
        n_y0 = int((target_values == 0).sum())
        n_y1 = int((target_values == 1).sum())
        scale = n_y0 / n_y1

        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale,
            n_estimators=500,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )
        self._model.fit(features, target_values)

        joblib.dump(self._model, self._MODEL_PATH)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            self._model = joblib.load(self._MODEL_PATH)

        predictions = self._model.predict(features)
        return [int(p) for p in predictions]

    @staticmethod
    def _get_min_diff(row: pd.Series) -> float:
        fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        return ((fecha_o - fecha_i).total_seconds()) / 60