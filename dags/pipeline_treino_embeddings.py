"""DAG 3: Treino diário de embeddings + indexação em pgvector."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task

from mlops_pratica.models.embeddings.encode import encode_dataframe
from mlops_pratica.models.embeddings.index import index_embeddings
from mlops_pratica.storage.minio_io import features_text_path, read_parquet

logger = logging.getLogger(__name__)


@dag(
    dag_id="pipeline_treino_embeddings",
    description="Gera embeddings dos títulos HN e indexa em pgvector.",
    schedule="0 3 * * *",  # 03:00 diariamente
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "embeddings", "pgvector"],
)
def pipeline_treino_embeddings():
    @task
    def load_text() -> dict:
        df = read_parquet(features_text_path())
        if df.empty:
            raise ValueError("features_text vazio.")
        return {"path": features_text_path(), "n_rows": len(df)}

    @task
    def encode(info: dict) -> dict:
        df = read_parquet(info["path"])
        df_emb, run_id = encode_dataframe(df, text_col="title")
        # Não devolve o DF inteiro via XCom; salva referência via caminho temporário
        # Para didática: passa adiante o caminho do parquet original + run_id.
        # O index task recodifica em memória (idempotente).
        return {"path": info["path"], "run_id": run_id}

    @task
    def index(info: dict) -> int:
        df = read_parquet(info["path"])
        df_emb, _ = encode_dataframe(df, text_col="title")
        return index_embeddings(df_emb, run_id=info["run_id"])

    info = load_text()
    enc = encode(info)
    index(enc)


pipeline_treino_embeddings()
