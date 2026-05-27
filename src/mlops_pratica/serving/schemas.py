"""Pydantic schemas para FastAPI."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    title: str = Field(..., description="Título do post HN")
    url: Optional[str] = Field(None, description="URL externa (opcional)")
    by_author: str = Field("unknown", description="Autor")
    hour: int = Field(..., ge=0, le=23)
    weekday: int = Field(..., ge=0, le=6)


class PredictResponse(BaseModel):
    probability_vai_bombar: float
    prediction: int
    model_name: str
    model_stage: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=50)


class SearchHit(BaseModel):
    id: int
    title: str
    url: Optional[str]
    score: Optional[int]
    by_author: Optional[str]
    similarity: float


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
