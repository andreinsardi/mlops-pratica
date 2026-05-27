"""Promoção do melhor modelo no MLflow Model Registry.

Regra: se a melhor versão recém-treinada tiver AUC > AUC da versão atual
em Production (ou nenhum em Production), promove para Staging.

Sobre o Model Registry
----------------------
O Registry é o "cadastro" de modelos do MLflow. Para cada nome (ex:
"hn_classifier"), ele guarda:
- VERSÕES (1, 2, 3, ...) — cada vez que `log_model` é chamado com esse
  nome, uma versão nova nasce.
- ESTÁGIOS por versão: None / Staging / Production / Archived. Servem
  como "etiqueta" para o consumidor saber qual versão usar.

No MLflow 2.x, as transições são feitas via `transition_model_version_stage`.
Esse método FOI DEPRECADO no MLflow 3 em favor de "aliases" (rótulos
livres como @champion, @challenger), porque os estágios fixos eram
limitantes. Como rodamos MLflow 2.16 neste projeto, ainda usamos a API
de estágios — o conceito didático é o mesmo.

Política aqui implementada (didática)
-------------------------------------
"Sempre promove a última versão para Staging e arquiva as anteriores."
- Simples de demonstrar em aula.
- NÃO compara métricas com a versão atual.
- NÃO exige aprovação humana.

Política recomendada em produção real
-------------------------------------
1. Comparar a métrica de negócio (não só AUC) da nova versão com a
   versão atual em Production.
2. Exigir margem mínima (ex: AUC novo - AUC antigo > 0.01) para evitar
   "promoção de ruído estatístico".
3. Promover para Staging primeiro, rodar testes de shadow / A-B.
4. Aprovação humana (gate manual) antes de virar Production.
5. Manter rollback fácil (não arquivar a versão anterior imediatamente).
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
    # `MlflowClient` é a API de baixo nível do MLflow para operações
    # administrativas (CRUD em runs, modelos, versões, tags). Funciona
    # contra qualquer backend (file, sqlite, postgres) — só depende do
    # tracking URI configurado acima.
    client = MlflowClient()

    # Query simples no Registry. A sintaxe `name='X'` é um mini-DSL do MLflow.
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logger.warning("Nenhuma versão encontrada para %s", model_name)
        return None

    # `version` é string ("1", "2", ...) — precisa converter para int
    # para ordenação numérica correta ("10" < "2" lexicograficamente).
    last = max(versions, key=lambda v: int(v.version))

    # `transition_model_version_stage`:
    # - move a versão indicada para o estágio alvo;
    # - `archive_existing_versions=True` arquiva automaticamente qualquer
    #   outra versão que estiver no MESMO estágio (mantém só UMA por
    #   estágio).
    # Reforço: esse método está DEPRECADO no MLflow 3 (usar aliases),
    # mas funciona normalmente no 2.16 que usamos aqui.
    client.transition_model_version_stage(
        name=model_name,
        version=last.version,
        stage=target_stage,
        archive_existing_versions=True,
    )
    logger.info("Modelo %s v%s promovido para %s", model_name, last.version, target_stage)
    return {"name": model_name, "version": last.version, "stage": target_stage}
