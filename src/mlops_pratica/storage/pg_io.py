"""I/O em Postgres + pgvector via SQLAlchemy."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from mlops_pratica.config import settings

logger = logging.getLogger(__name__)


def get_engine() -> Engine:
    return create_engine(settings.pg_uri, pool_pre_ping=True, future=True)


def _vector_literal(vec: Iterable[float]) -> str:
    """Serializa um vetor como literal pgvector: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def upsert_embeddings(df: pd.DataFrame, model_name: str, run_id: str) -> int:
    """Upsert em batch na tabela `embeddings`.

    Espera colunas: id, title, url, score, by_author, created_ts, embedding (np.ndarray).
    """
    if df.empty:
        logger.info("Nada para upsert em embeddings.")
        return 0

    eng = get_engine()
    sql = text("""
        INSERT INTO embeddings
            (id, title, url, score, by_author, created_ts,
             embedding, model_name, model_run_id)
        VALUES
            (:id, :title, :url, :score, :by_author, :created_ts,
             CAST(:embedding AS vector), :model_name, :model_run_id)
        ON CONFLICT (id) DO UPDATE SET
            title        = EXCLUDED.title,
            url          = EXCLUDED.url,
            score        = EXCLUDED.score,
            by_author    = EXCLUDED.by_author,
            created_ts   = EXCLUDED.created_ts,
            embedding    = EXCLUDED.embedding,
            model_name   = EXCLUDED.model_name,
            model_run_id = EXCLUDED.model_run_id,
            indexed_at   = NOW();
    """)

    rows = []
    for r in df.itertuples(index=False):
        emb = r.embedding
        if isinstance(emb, np.ndarray):
            emb_lit = _vector_literal(emb.tolist())
        else:
            emb_lit = _vector_literal(emb)
        rows.append(
            {
                "id": int(r.id),
                "title": r.title,
                "url": r.url if pd.notna(r.url) else None,
                "score": int(r.score) if pd.notna(r.score) else None,
                "by_author": r.by_author if pd.notna(r.by_author) else None,
                "created_ts": r.created_ts,
                "embedding": emb_lit,
                "model_name": model_name,
                "model_run_id": run_id,
            }
        )

    with eng.begin() as conn:
        conn.execute(sql, rows)
    logger.info("Upsert de %d embeddings concluído.", len(rows))
    return len(rows)


def search_similar(
    query_vec: np.ndarray, k: int = 10
) -> list[dict]:
    """Busca semântica por similaridade cosseno."""
    eng = get_engine()
    vec_lit = _vector_literal(query_vec.tolist())
    sql = text(f"""
        SELECT id, title, url, score, by_author, created_ts,
               1 - (embedding <=> CAST(:vec AS vector)) AS similarity
        FROM embeddings
        ORDER BY embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """)
    with eng.connect() as conn:
        result = conn.execute(sql, {"vec": vec_lit, "k": k})
        rows = [dict(r._mapping) for r in result]
    return rows
