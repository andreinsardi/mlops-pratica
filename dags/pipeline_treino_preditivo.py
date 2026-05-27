"""DAG 2: Treino diário do classificador 'vai_bombar'.

Carrega features tabulares, treina sklearn (RF) + XGBoost, registra
o vencedor no MLflow Model Registry e promove para Staging.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task

from mlops_pratica.config import settings
from mlops_pratica.models.preditivo.register import promote_best_version
from mlops_pratica.models.preditivo.train import train_and_register
from mlops_pratica.storage.minio_io import features_tabular_path, read_parquet

logger = logging.getLogger(__name__)


@dag(
    dag_id="pipeline_treino_preditivo",
    description="Treina classificador HN (vai_bombar) e promove no MLflow Registry.",
    schedule="0 2 * * *",  # 02:00 diariamente
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "treino", "preditivo"],
)
def pipeline_treino_preditivo():
    @task
    def load_features() -> dict:
        df = read_parquet(features_tabular_path())
        if len(df) < 100:
            raise ValueError(f"Poucos exemplos para treino: {len(df)}")
        return {"n_rows": len(df), "path": features_tabular_path()}

    @task
    def train_and_register_models(info: dict) -> dict:
        df = read_parquet(info["path"])
        results = train_and_register(df, register_name=settings.model_name_registry)
        return {
            "winner": results["winner"],
            "winner_metrics": results[results["winner"]]["metrics"],
        }

    @task
    def promote(results: dict) -> dict:
        info = promote_best_version(settings.model_name_registry, target_stage="Staging")
        return {"promoted": info, "winner": results["winner"]}

    info = load_features()
    results = train_and_register_models(info)
    promote(results)


pipeline_treino_preditivo()
