"""Configurações centrais lidas de variáveis de ambiente.

Toda a stack roda em Docker; as variáveis são definidas em docker-compose.yml
e .env. Para uso local (testes), há valores default sensatos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # ---------------------------------------------------------------- MLflow
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow_experiment_preditivo: str = os.getenv(
        "MLFLOW_EXPERIMENT_PREDITIVO", "hn_classifier"
    )
    mlflow_experiment_embeddings: str = os.getenv(
        "MLFLOW_EXPERIMENT_EMBEDDINGS", "hn_embeddings"
    )

    # ---------------------------------------------------------------- MinIO/S3
    s3_endpoint_url: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    bucket_raw: str = "raw"
    bucket_curated: str = "curated"
    bucket_features: str = "features"

    # ---------------------------------------------------------------- Postgres app (pgvector)
    pg_host: str = os.getenv("PG_HOST", "postgres")
    pg_port: int = int(os.getenv("PG_PORT", "5432"))
    pg_db: str = os.getenv("PG_DB", "app")
    pg_user: str = os.getenv("PG_USER", "app")
    pg_password: str = os.getenv("PG_PASSWORD", "app")

    # ---------------------------------------------------------------- HackerNews
    hn_api_base: str = os.getenv("HN_API_BASE", "https://hacker-news.firebaseio.com/v0")
    hn_top_n: int = int(os.getenv("HN_TOP_N", "100"))

    # ---------------------------------------------------------------- Embeddings
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # ---------------------------------------------------------------- Modelo registrado
    model_name_registry: str = os.getenv("MODEL_NAME", "hn_classifier")
    model_stage: str = os.getenv("MODEL_STAGE", "Production")

    @property
    def pg_uri(self) -> str:
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


settings = Settings()
