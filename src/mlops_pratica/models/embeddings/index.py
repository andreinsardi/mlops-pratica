"""Indexação dos embeddings em pgvector via storage.pg_io.

Camada finíssima que delega o trabalho real para `storage.pg_io.upsert_embeddings`.
Por que então existir como módulo separado?
- Mantém a separação de responsabilidades clara:
    models/embeddings/encode.py  -> CRIA vetores
    models/embeddings/index.py   -> COLOCA vetores no índice (pgvector)
    storage/pg_io.py             -> sabe FALAR com o Postgres
- Se um dia trocarmos o índice (ex: para FAISS, Qdrant, Milvus), só este
  arquivo muda — os DAGs do Airflow continuam chamando `index_embeddings`.
- Acoplamento entre etapas do pipeline = baixo; coesão dentro de cada
  módulo = alta. É o princípio de "ports & adapters" aplicado em pequena
  escala.
"""

from __future__ import annotations

import logging

import pandas as pd

from mlops_pratica.config import settings
from mlops_pratica.storage.pg_io import upsert_embeddings

logger = logging.getLogger(__name__)


def index_embeddings(df: pd.DataFrame, run_id: str) -> int:
    """Upsert dos embeddings na tabela `embeddings`.

    `run_id` vem do `encode_dataframe` (run MLflow do encoder) e é
    persistido no banco para rastreabilidade (qual execução criou
    cada vetor).
    """
    if df.empty:
        logger.info("Sem embeddings para indexar.")
        return 0
    n = upsert_embeddings(df, model_name=settings.embedding_model_name, run_id=run_id)
    logger.info("Indexados %d vetores em pgvector.", n)
    return n
