"""Codificação de textos em vetores densos com sentence-transformers.

Cada execução é logada no MLflow como um run do experimento de embeddings,
registrando: nome do modelo, dimensionalidade, quantidade de itens vetorizados,
amostra de inputs/outputs (artifact).
"""

from __future__ import annotations

import logging
from typing import Iterable

import mlflow
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from mlops_pratica.config import settings

logger = logging.getLogger(__name__)

_model_cache: SentenceTransformer | None = None


def get_encoder() -> SentenceTransformer:
    """Lazy-load do modelo. Reusa instância dentro do processo."""
    global _model_cache
    if _model_cache is None:
        logger.info("Carregando encoder %s", settings.embedding_model_name)
        _model_cache = SentenceTransformer(settings.embedding_model_name)
    return _model_cache


def encode_texts(texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
    model = get_encoder()
    vecs = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs


def encode_dataframe(df: pd.DataFrame, text_col: str = "title") -> tuple[pd.DataFrame, str]:
    """Adiciona coluna `embedding` ao dataframe e loga no MLflow.

    Retorna (df_com_embedding, run_id).
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_embeddings)

    df = df.copy()

    with mlflow.start_run(run_name="encode_titles") as run:
        mlflow.set_tag("modelo", settings.embedding_model_name)
        mlflow.log_param("embedding_model", settings.embedding_model_name)
        mlflow.log_param("embedding_dim", settings.embedding_dim)
        mlflow.log_param("n_inputs", len(df))

        vecs = encode_texts(df[text_col].tolist())
        df["embedding"] = list(vecs)

        # Amostras como artifact (sanity check)
        sample = df[["id", text_col]].head(10).to_dict(orient="records")
        mlflow.log_dict({"sample_inputs": sample}, "samples.json")
        mlflow.log_metric("n_vectors", len(vecs))
        mlflow.log_metric("vector_dim", vecs.shape[1])

        logger.info("Encode concluído: %d vetores (run_id=%s)", len(vecs), run.info.run_id)
        return df, run.info.run_id
