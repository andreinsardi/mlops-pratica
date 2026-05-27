"""Promoção do melhor modelo no MLflow Model Registry.

Regra: se a melhor versão recém-treinada tiver AUC > AUC da versão atual
em Production (ou nenhum em Production), promove para Staging.
"""

from __future__ import annotations

import logging

import mlflow
from mlflow.tracking import MlflowClient

from mlops_pratica.config import settings

logger = logging.getLogger(__name__)


def promote_best_version(model_name: str, target_stage: str = "Staging") -> dict | None:
    """Promove a última versão registrada para o stage indicado.

    Política didática: a versão mais recente vai para Staging.
    Em prod real, comparar métricas e exigir aprovação humana.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logger.warning("Nenhuma versão encontrada para %s", model_name)
        return None

    # Última versão por número
    last = max(versions, key=lambda v: int(v.version))

    client.transition_model_version_stage(
        name=model_name,
        version=last.version,
        stage=target_stage,
        archive_existing_versions=True,
    )
    logger.info("Modelo %s v%s promovido para %s", model_name, last.version, target_stage)
    return {"name": model_name, "version": last.version, "stage": target_stage}
