from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.causal import estimate_randomized_uplift
from src.recommender import ItemRecommender

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="AI Growth & Personalization API", version="1.0.0")


@lru_cache
def recommender() -> ItemRecommender:
    return ItemRecommender().fit(pd.read_csv(ROOT / "data/events.csv"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommendations/{customer_id}")
def recommendations(customer_id: str, n: int = 5):
    try:
        return {"customer_id": customer_id, "recommendations": recommender().recommend(customer_id, n)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/experiment/uplift")
def experiment_uplift():
    result = estimate_randomized_uplift(pd.read_csv(ROOT / "data/experiment.csv"))
    return result.__dict__

