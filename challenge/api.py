import os

import fastapi
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List

from challenge.model import DelayModel

app = fastapi.FastAPI()
model = DelayModel()

# Load pre-trained model or train from data on import
if os.path.exists(DelayModel._MODEL_PATH):
    import joblib
    model._model = joblib.load(DelayModel._MODEL_PATH)
else:
    _data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "data.csv"
    )
    if os.path.exists(_data_path):
        _data = pd.read_csv(_data_path)
        _features, _target = model.preprocess(_data, target_column="delay")
        model.fit(_features, _target)
        del _data, _features, _target


class FlightInput(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class PredictInput(BaseModel):
    flights: List[FlightInput]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(data: PredictInput) -> dict:
    for flight in data.flights:
        if flight.MES < 1 or flight.MES > 12:
            raise HTTPException(status_code=400, detail="Invalid MES value")
        if flight.TIPOVUELO not in ("I", "N"):
            raise HTTPException(status_code=400, detail="Invalid TIPOVUELO value")

    flights_df = pd.DataFrame([f.dict() for f in data.flights])
    features = model.preprocess(flights_df)
    predictions = model.predict(features)
    return {"predict": predictions}