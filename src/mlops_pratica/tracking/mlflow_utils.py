"""Utilitários comuns de MLflow (set experiment, load model do registry)."""

from __future__ import annotations

import logging

import mlflow

from mlops_pratica.config import settings

logger = logging.getLogger(__name__)


def setup_mlflow(experiment: str) -> None:
    """Atalho para configurar tracking URI + experimento ativo de uma vez.

    Útil em entry-points (scripts CLI, DAGs Airflow) que sempre fazem
    esses dois passos antes de qualquer `mlflow.start_run()`.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment)


def load_production_model(model_name: str, stage: str = "Production"):
    """Carrega modelo do Registry no stage indicado.

    Se nenhum modelo estiver em `stage`, tenta Staging; depois, última versão.

    Por que esse fallback Production -> Staging -> latest?
    ------------------------------------------------------
    Em DESENVOLVIMENTO, é comum não ter ninguém em Production ainda
    (acabou de treinar o primeiro modelo). Sem fallback, a API quebra
    no startup e o aluno fica com erro 500 sem entender o porquê.

    Em PRODUÇÃO real, esse fallback deveria ser MAIS rigoroso:
    - Talvez só Production (falhar alto se não houver é o correto,
      para evitar servir um modelo experimental por acidente);
    - Ou Production -> alerta no PagerDuty + falha;
    - Nunca cair para "latest" silenciosamente.

    Aqui é didático: privilegia "funciona em sala" sobre rigor.

    URI scheme `models:/`
    ---------------------
    O MLflow usa um schema próprio para resolver versões do Registry:
        models:/<nome>/<stage_ou_versão_ou_alias>
    Ex.: `models:/hn_classifier/Production`
         `models:/hn_classifier/3`
         `models:/hn_classifier/latest`
    `mlflow.sklearn.load_model` deserializa o pipeline completo (preprocessor
    + estimador) — mesmo que o `clf` interno seja XGBClassifier, ele foi
    salvo dentro de um Pipeline sklearn, então o loader sklearn funciona.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # Lista ordenada por preferência. Itera tentando cada URI até um funcionar.
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
            # Log de WARNING (não ERROR): tentar Staging quando não há
            # Production é esperado; só vira problema se tudo falhar.
            last_err = exc
            logger.warning("Falha em %s: %s", uri, exc)
    # Se chegamos aqui, NENHUM stage existe — agora sim é erro real.
    raise RuntimeError(f"Nenhum modelo encontrado para {model_name}: {last_err}")
