import os
import threading
import time

import fastapi
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from challenge.model import DelayModel

MAX_FLIGHTS_PER_REQUEST = 1000


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket por proceso — mitiga DoS (certificacion DevSecOps F2)."""

    def __init__(self, app, rate_per_sec: float = 5.0, burst: int = 10):
        super().__init__(app)
        self._rate = rate_per_sec
        self._burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    async def dispatch(self, request, call_next):
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._burst,
                self._tokens + (now - self._last) * self._rate,
            )
            self._last = now
            if self._tokens < 1:
                return JSONResponse(
                    {"detail": "Too Many Requests"}, status_code=429
                )
            self._tokens -= 1
        return await call_next(request)


app = fastapi.FastAPI()
app.add_middleware(RateLimitMiddleware, rate_per_sec=5.0, burst=10)
model = DelayModel()

# Load pre-trained model or train from data automatically
if os.path.exists(DelayModel._MODEL_PATH):
    model._load()
else:
    _data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "data.csv"
    )
    if os.path.exists(_data_path):
        model.train_from_csv(_data_path)


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
    if len(data.flights) > MAX_FLIGHTS_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"max {MAX_FLIGHTS_PER_REQUEST} flights per request",
        )
    for flight in data.flights:
        if flight.MES < 1 or flight.MES > 12:
            raise HTTPException(status_code=400, detail="Invalid MES value")
        if flight.TIPOVUELO not in ("I", "N"):
            raise HTTPException(status_code=400, detail="Invalid TIPOVUELO value")

    flights_list = [f.dict() for f in data.flights]
    features = model.preprocess_api(flights_list)
    predictions = model.predict(features)
    return {"predict": predictions}