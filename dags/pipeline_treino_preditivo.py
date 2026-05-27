"""DAG 2: Treino diário do classificador 'vai_bombar'.

Esta DAG consome o parquet de features tabulares produzido pela DAG 1
(`pipeline_ingestao`) e treina dois modelos de baseline (Random Forest do
scikit-learn e XGBoost), comparando-os por métrica. O vencedor é registrado
no MLflow Model Registry e promovido para o stage "Staging", de onde a
aplicação de serving (FastAPI) pode carregá-lo pelo alias.

Fluxo (3 tasks, todas em série):
  1) load_features              -> lê o parquet e valida volume mínimo
  2) train_and_register_models  -> treina, loga métricas no MLflow Tracking,
                                   versiona modelo no MLflow Registry
  3) promote                    -> move a versão recém-criada para "Staging"

Integração com MLflow (dois subsistemas distintos do mesmo servidor):
  - Tracking: armazena RUNS com parâmetros, métricas e artefatos. Cada
    `mlflow.start_run()` cria um run com ID único e histórico imutável.
  - Registry: catálogo de MODELOS NOMEADOS com versões e stages
    (None/Staging/Production/Archived). É o que a produção consulta para
    saber "qual modelo deve estar servindo agora".

Decisões de design didáticas:
  - Schedule cron `0 2 * * *` = "minuto 0, hora 2, todo dia, todo mês, todo
    dia da semana" => 02:00 diariamente. Roda à noite para não competir com
    a ingestão horária do dia útil.
  - Separar `train` de `promote` em tasks distintas permite: (a) re-executar
    APENAS a promoção sem treinar de novo (custo zero), (b) observar e
    auditar cada etapa isoladamente nos logs do Airflow, (c) aplicar
    políticas de retry diferentes para cada uma no futuro.
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
    # Cron: "minuto hora dia-do-mes mes dia-da-semana".
    # "0 2 * * *" = todo dia às 02:00. Horário escolhido para rodar depois
    # que a ingestão noturna acumulou dados frescos do dia anterior.
    schedule="0 2 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,            # sem backfill (mesma justificativa da DAG 1)
    max_active_runs=1,        # nunca treinar dois modelos em paralelo
    default_args={
        "owner": "mlops",
        # Apenas 1 retry: treino é caro (cpu/memória). Se falhar 2 vezes,
        # provavelmente é bug ou problema de dado — não vale insistir.
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "treino", "preditivo"],
)
def pipeline_treino_preditivo():
    @task
    def load_features() -> dict:
        """Lê o parquet de features e aplica o "gate" de volume mínimo.

        `MIN_TRAIN_SAMPLES` (variavel de ambiente / settings) protege contra
        treino prematuro: sem dados suficientes, o modelo aprende ruído e
        métricas viram aleatórias, poluindo o Registry com versões ruins.

        Aqui usamos um valor pequeno (didático). Em produção real, gates
        típicos exigem milhares de amostras e checagens adicionais (drift,
        balanceamento de classes, frescor dos dados, etc.).

        Retorna apenas metadados leves (n_rows e path) via XCom — o DataFrame
        em si será relido na próxima task a partir do path. Padrão para
        evitar serializar dados grandes pelo XCom backend.
        """
        df = read_parquet(features_tabular_path())
        if len(df) < settings.min_train_samples:
            raise ValueError(
                f"Poucos exemplos para treino: {len(df)} "
                f"(mínimo configurado MIN_TRAIN_SAMPLES={settings.min_train_samples})"
            )
        return {"n_rows": len(df), "path": features_tabular_path()}

    @task
    def train_and_register_models(info: dict) -> dict:
        """Treina RF + XGBoost, escolhe vencedor e registra no MLflow.

        A função `train_and_register` (em `models/preditivo/train.py`):
          - abre um run no MLflow Tracking (`mlflow.start_run`)
          - loga hiperparâmetros (`mlflow.log_param`) e métricas
            (`mlflow.log_metric` para AUC, F1, etc.)
          - serializa o melhor modelo com `mlflow.sklearn.log_model(...,
            registered_model_name=...)`, que cria automaticamente uma nova
            VERSÃO no Model Registry sob o nome `settings.model_name_registry`.

        Retornamos apenas o nome do vencedor + métricas (pequeno) para a
        próxima task — nunca o objeto sklearn em si.
        """
        df = read_parquet(info["path"])
        results = train_and_register(df, register_name=settings.model_name_registry)
        return {
            "winner": results["winner"],
            "winner_metrics": results[results["winner"]]["metrics"],
        }

    @task
    def promote(results: dict) -> dict:
        """Promove a versão mais recente para o stage "Staging".

        Política didática implementada em `promote_best_version`: pega a
        ÚLTIMA versão registrada e a move para "Staging", sem comparar com
        a versão atualmente em produção.

        Em produção real, a estratégia recomendada é: comparar a métrica da
        nova versão com a da que já está em Staging/Production e só promover
        se houver melhoria significativa (ex.: AUC > AUC_atual + epsilon),
        evitando degradação silenciosa por overfitting ou drift de dados.
        """
        info = promote_best_version(settings.model_name_registry, target_stage="Staging")
        return {"promoted": info, "winner": results["winner"]}

    # Grafo linear: cada task depende somente do retorno da anterior.
    # Não há paralelismo aqui — treino e promoção são naturalmente sequenciais.
    info = load_features()
    results = train_and_register_models(info)
    promote(results)


pipeline_treino_preditivo()
