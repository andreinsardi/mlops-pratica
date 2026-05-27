"""Utilitários comuns de MLflow (set experiment, load model do registry)."""

from __future__ import annotations

import logging

import mlflow

from mlops_pratica.config import settings

logger = logging.getLogger(__name__)


def setup_mlflow(experiment: str) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment)


def load_production_model(model_name: str, stage: str = "Production"):
    """Carrega modelo do Registry no stage indicado.

    Se nenhum modelo estiver em `stage`, tenta Staging; depois, última versão.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri_candidates = [
        f"models:/{model_name}/{stage}",
        f"models:/{model_name}/Staging",
        f"models:/{model_name}/latest",
    ]
    last_err: Exception | None = None
    for uri in uri_candidates:
        try:
            model = mlflow.sklearn.load_model(uri)
            logger.info("Modelo carregado de %s", uri)
            return model
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Falha em %s: %s", uri, exc)
    raise RuntimeError(f"Nenhum modelo encontrado para {model_name}: {last_err}")
