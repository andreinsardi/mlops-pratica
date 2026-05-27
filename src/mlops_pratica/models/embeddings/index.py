"""Indexação dos embeddings em pgvector via storage.pg_io."""

from __future__ import annotations

import logging

import pandas as pd

from mlops_pratica.config import settings
from mlops_pratica.storage.pg_io import upsert_embeddings

logger = logging.getLogger(__name__)


def index_embeddings(df: pd.DataFrame, run_id: str) -> int:
    """Upsert dos embeddings na tabela `embeddings`."""
    if df.empty:
        logger.info("Sem embeddings para indexar.")
        return 0
    n = upsert_embeddings(df, model_name=settings.embedding_model_name, run_id=run_id)
    logger.info("Indexados %d vetores em pgvector.", n)
    return n
