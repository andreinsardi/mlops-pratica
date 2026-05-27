"""DAG 1: Ingestão horária do HackerNews.

Fluxo:
  1) extract  -> chama API HN e baixa top N
  2) raw      -> grava parquet em s3://raw/hn/dt=YYYY-MM-DD/hr=HH/items.parquet
  3) curated  -> normaliza + dedup -> s3://curated/hn/stories.parquet (append-merge)
  4) features -> gera features tabular + text para os pipes de treino
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
from airflow.decorators import dag, task

from mlops_pratica.config import settings
from mlops_pratica.features.tabular import build_tabular_features
from mlops_pratica.features.text import build_text_features
from mlops_pratica.ingestion.extractor import fetch_top_stories
from mlops_pratica.ingestion.hn_client import HackerNewsClient
from mlops_pratica.storage.minio_io import (
    curated_path,
    features_tabular_path,
    features_text_path,
    raw_path,
    read_parquet,
    write_parquet,
)

logger = logging.getLogger(__name__)


@dag(
    dag_id="pipeline_ingestao",
    description="Ingestão horária do HackerNews para data lake MinIO.",
    schedule="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["mlops", "ingestao", "hackernews"],
)
def pipeline_ingestao():
    @task
    def extract(**ctx) -> str:
        """Extrai top stories e grava raw parquet."""
        logical = ctx["logical_date"]
        dt = logical.strftime("%Y-%m-%d")
        hr = logical.strftime("%H")

        client = HackerNewsClient()
        items = fetch_top_stories(client=client, limit=settings.hn_top_n)
        if not items:
            raise ValueError("Nenhum item extraído.")

        df = pd.DataFrame(items)
        path = raw_path(dt=dt, hr=hr)
        write_parquet(df, path)
        return path

    @task
    def curate(raw_s3_path: str) -> str:
        """Normaliza e mescla com camada curated (dedup por id)."""
        df_new = read_parquet(raw_s3_path)

        # Tenta ler curated atual; se não existir, começa do zero
        try:
            df_cur = read_parquet(curated_path())
        except Exception:  # noqa: BLE001
            df_cur = pd.DataFrame()

        df_all = pd.concat([df_cur, df_new], ignore_index=True)
        # Mantém versão com maior score (último snapshot por id)
        df_all = (
            df_all.sort_values("score", ascending=False, na_position="last")
            .drop_duplicates(subset=["id"], keep="first")
            .reset_index(drop=True)
        )
        write_parquet(df_all, curated_path())
        logger.info("Curated: %d stories acumuladas.", len(df_all))
        return curated_path()

    @task
    def build_features_tabular(curated_s3_path: str) -> str:
        df = read_parquet(curated_s3_path)
        feats = build_tabular_features(df)
        write_parquet(feats, features_tabular_path())
        return features_tabular_path()

    @task
    def build_features_text(curated_s3_path: str) -> str:
        df = read_parquet(curated_s3_path)
        feats = build_text_features(df)
        write_parquet(feats, features_text_path())
        return features_text_path()

    raw = extract()
    curated = curate(raw)
    build_features_tabular(curated)
    build_features_text(curated)


pipeline_ingestao()
