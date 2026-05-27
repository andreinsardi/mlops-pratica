"""Engenharia de features tabulares para classificador 'vai bombar' (score>=100)."""

from __future__ import annotations

from urllib.parse import urlparse

import numpy as np
import pandas as pd

SCORE_THRESHOLD = 100  # alvo binário: vai_bombar = (score >= 100)


def _extract_domain(url: object) -> str:
    if not isinstance(url, str) or not url:
        return "no_url"
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.split(":")[0] or "no_url"
    except Exception:
        return "no_url"


def build_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe stories curadas e devolve dataset pronto para treino.

    Saída tem as colunas:
      id, title_len, n_words_title, hour, weekday, has_url, has_question,
      domain, by_author, type, score, vai_bombar
    """
    df = df.copy()

    # Garante tipos
    df = df[df["type"].fillna("") == "story"]
    df = df.dropna(subset=["title", "time", "score"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["title"] = df["title"].astype(str)
    df["created_ts"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["created_ts"])

    # Features
    df["title_len"] = df["title"].str.len()
    df["n_words_title"] = df["title"].str.split().str.len()
    df["hour"] = df["created_ts"].dt.hour
    df["weekday"] = df["created_ts"].dt.weekday
    df["has_url"] = df["url"].fillna("").astype(str).str.len().gt(0).astype(int)
    df["has_question"] = df["title"].str.contains(r"\?$", regex=True).astype(int)
    df["domain"] = df["url"].apply(_extract_domain)
    df["by_author"] = df["by"].fillna("unknown")

    # Target
    df["vai_bombar"] = (df["score"] >= SCORE_THRESHOLD).astype(int)

    keep = [
        "id",
        "title",
        "title_len",
        "n_words_title",
        "hour",
        "weekday",
        "has_url",
        "has_question",
        "domain",
        "by_author",
        "type",
        "url",
        "score",
        "created_ts",
        "vai_bombar",
    ]
    return df[keep].reset_index(drop=True)


def feature_columns() -> list[str]:
    """Colunas (X) usadas no treino do modelo preditivo."""
    return [
        "title_len",
        "n_words_title",
        "hour",
        "weekday",
        "has_url",
        "has_question",
        "domain",
        "by_author",
    ]


def target_column() -> str:
    return "vai_bombar"
