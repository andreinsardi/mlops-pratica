"""FastAPI - endpoints /predict e /search.

GET  /health   -> liveness
POST /predict  -> classifica HN story (vai_bombar)
POST /search   -> busca semântica em pgvector
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from mlops_pratica.config import settings
from mlops_pratica.features.tabular import _extract_domain
from mlops_pratica.models.embeddings.encode import encode_texts
from mlops_pratica.serving.schemas import (
    PredictRequest,
    PredictResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from mlops_pratica.storage.pg_io import search_similar
from mlops_pratica.tracking.mlflow_utils import load_production_model

logger = logging.getLogger(__name__)

# Estado global carregado em startup
state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Carrega modelo do Registry na inicialização."""
    try:
        state["model"] = load_production_model(
            settings.model_name_registry, settings.model_stage
        )
        logger.info("Modelo %s/%s carregado", settings.model_name_registry, settings.model_stage)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Não foi possível carregar modelo no startup: %s", exc)
        state["model"] = None
    yield
    state.clear()


app = FastAPI(
    title="MLOps Prática - HackerNews",
    description="API de predição (vai_bombar) e busca semântica (pgvector).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": state.get("model") is not None,
        "tracking_uri": settings.mlflow_tracking_uri,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = state.get("model")
    if model is None:
        # Tenta carregar tardiamente (caso ainda não houvesse modelo na startup)
        try:
            state["model"] = load_production_model(
                settings.model_name_registry, settings.model_stage
            )
            model = state["model"]
        except Exception as exc:
            raise HTTPException(503, f"Modelo indisponível: {exc}") from exc

    row = {
        "title_len": len(req.title),
        "n_words_title": len(req.title.split()),
        "hour": req.hour,
        "weekday": req.weekday,
        "has_url": int(bool(req.url)),
        "has_question": int(req.title.rstrip().endswith("?")),
        "domain": _extract_domain(req.url),
        "by_author": req.by_author,
    }
    df = pd.DataFrame([row])
    proba = float(model.predict_proba(df)[0, 1])
    return PredictResponse(
        probability_vai_bombar=proba,
        prediction=int(proba >= 0.5),
        model_name=settings.model_name_registry,
        model_stage=settings.model_stage,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    vec = encode_texts([req.query])[0]
    rows = search_similar(vec, k=req.k)
    hits = [
        SearchHit(
            id=r["id"],
            title=r["title"],
            url=r["url"],
            score=r["score"],
            by_author=r["by_author"],
            similarity=float(r["similarity"]),
        )
        for r in rows
    ]
    return SearchResponse(query=req.query, hits=hits)
