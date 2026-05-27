"""Preparação de features textuais (títulos) para o pipe de embeddings."""

from __future__ import annotations

import pandas as pd


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna dataframe com colunas mínimas para encoder de embeddings.

    Saída: id, title, url, score, by_author, created_ts
    """
    df = df.copy()
    df = df[df["type"].fillna("") == "story"]
    df = df.dropna(subset=["title", "time"])
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 0]

    df["created_ts"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df["by_author"] = df["by"].fillna("unknown")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["url"] = df["url"].fillna("")

    keep = ["id", "title", "url", "score", "by_author", "created_ts"]
    return df[keep].drop_duplicates(subset=["id"]).reset_index(drop=True)
