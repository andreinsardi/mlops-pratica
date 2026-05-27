"""DAG 1: Ingestão horária do HackerNews.

Esta é a primeira DAG do projeto MLOps Prática. Sua responsabilidade é
alimentar continuamente o data lake (MinIO, compatível com S3) com dados
brutos da API pública do HackerNews, transformando-os em camadas cada vez
mais limpas até produzirem as tabelas de features consumidas pelas DAGs
de treino (DAG 2 e DAG 3).

Fluxo (arquitetura medalhão simplificada raw -> curated -> features):
  1) extract  -> chama API HN e baixa top N stories
  2) raw      -> grava parquet em s3://raw/hn/dt=YYYY-MM-DD/hr=HH/items.parquet
                 (particionamento Hive-style: facilita leitura incremental e
                  permite reprocessar uma hora/dia específico sem tocar no resto)
  3) curated  -> normaliza + dedup -> s3://curated/hn/stories.parquet
                 (append-merge: junta novos itens ao histórico e remove duplicatas)
  4) features -> gera features tabulares (DAG 2) + text (DAG 3) em paralelo

Conceitos didáticos demonstrados aqui:
  - TaskFlow API do Airflow (decorators `@dag` e `@task`): forma moderna de
    declarar pipelines em Python puro, sem precisar instanciar Operators
    clássicos (PythonOperator, BashOperator). O retorno de uma `@task` vira
    automaticamente input da próxima via XCom, deixando o código mais limpo
    e parecido com código Python "normal".
  - Camadas medalhão (raw -> curated -> features) com responsabilidades
    bem separadas, cada uma com seu prefixo de storage.
  - Idempotência: cada hora reprocessa o curated inteiro (dedup garante
    consistência mesmo se a mesma execução rodar duas vezes).
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
    # `@hourly` é equivalente ao cron "0 * * * *" (no minuto 0 de cada hora).
    schedule="@hourly",
    start_date=datetime(2025, 1, 1),
    # `catchup=False`: não faz backfill automático das execuções "perdidas"
    # entre `start_date` e hoje. Para este projeto didático isso é o ideal,
    # pois não queremos disparar centenas de runs históricas ao subir o DAG.
    catchup=False,
    # `max_active_runs=1`: garante que duas execuções da mesma DAG nunca
    # rodem em paralelo. Importante porque o passo `curate` faz append-merge
    # no mesmo arquivo curated — execuções simultâneas causariam race condition.
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        # Política de retry para tolerar falhas transientes (rede instável,
        # timeout temporário da API do HN, etc.). 2 tentativas extras com
        # 2 minutos entre elas costuma resolver intermitências leves sem
        # mascarar bugs reais (que falhariam consistentemente).
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["mlops", "ingestao", "hackernews"],
)
def pipeline_ingestao():
    @task
    def extract(**ctx) -> str:
        """Extrai top stories da API HN e grava na camada raw em parquet.

        Usa o `logical_date` do contexto do Airflow (não `datetime.now()`)
        para garantir reprocessabilidade: se reexecutarmos esta run no futuro,
        ela escreverá na MESMA partição (dt/hr) original. Isso é fundamental
        para idempotência em pipelines de dados.

        O retorno (string com o caminho S3) é automaticamente serializado
        em XCom pelo TaskFlow API e fica disponível para a próxima task.
        Boa prática: passar APENAS referências leves (paths, IDs) via XCom,
        nunca DataFrames inteiros, pois o backend de XCom não é otimizado
        para dados grandes.
        """
        logical = ctx["logical_date"]
        dt = logical.strftime("%Y-%m-%d")
        hr = logical.strftime("%H")

        client = HackerNewsClient()
        items = fetch_top_stories(client=client, limit=settings.hn_top_n)
        if not items:
            # Falhar explícito é melhor que gravar parquet vazio: o Airflow
            # marca a task como failed e dispara a política de retry.
            raise ValueError("Nenhum item extraído.")

        df = pd.DataFrame(items)
        # Particionamento Hive-style "dt=YYYY-MM-DD/hr=HH". Esse formato é
        # reconhecido por engines como Spark, Trino e DuckDB, que conseguem
        # fazer "partition pruning" automático ao filtrar por data/hora.
        path = raw_path(dt=dt, hr=hr)
        write_parquet(df, path)
        return path

    @task
    def curate(raw_s3_path: str) -> str:
        """Normaliza o snapshot horário e mescla na camada curated (dedup por id).

        Recebe via XCom o caminho do parquet raw produzido pela task `extract`.
        O TaskFlow API converte automaticamente o return value anterior em
        argumento desta função — não precisamos manipular XCom manualmente.

        Estratégia de dedup: a API do HackerNews retorna o snapshot atual de
        cada story (score, descendants, etc.) mudando ao longo do tempo. O
        MESMO `id` aparece em várias execuções horárias com valores diferentes.
        Aqui decidimos manter a versão com MAIOR score por id (sort + drop
        duplicates keep="first"). Justificativa didática: queremos representar
        o "pico" de engajamento de cada story, que é o sinal mais informativo
        para a label `vai_bombar` do classificador da DAG 2.
        """
        df_new = read_parquet(raw_s3_path)

        # Tenta ler curated atual; se ainda não existir (primeira execução),
        # começa do zero com DataFrame vazio. Padrão "empty-or-existing" comum
        # em pipelines incrementais que precisam funcionar no dia zero.
        try:
            df_cur = read_parquet(curated_path())
        except Exception:  # noqa: BLE001
            df_cur = pd.DataFrame()

        df_all = pd.concat([df_cur, df_new], ignore_index=True)
        # Ordena por score desc e mantém a primeira ocorrência de cada id =>
        # equivale a "para cada id, manter o snapshot com maior score já visto".
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
        """Gera features numéricas/categóricas para o classificador (DAG 2).

        Consome a camada curated e produz `features_tabular.parquet`, que será
        lido pela DAG `pipeline_treino_preditivo` no horário 02:00.
        """
        df = read_parquet(curated_s3_path)
        feats = build_tabular_features(df)
        write_parquet(feats, features_tabular_path())
        return features_tabular_path()

    @task
    def build_features_text(curated_s3_path: str) -> str:
        """Gera features textuais (títulos limpos) para embeddings (DAG 3).

        Consome a camada curated e produz `features_text.parquet`, que será
        lido pela DAG `pipeline_treino_embeddings` no horário 03:00.
        """
        df = read_parquet(curated_s3_path)
        feats = build_text_features(df)
        write_parquet(feats, features_text_path())
        return features_text_path()

    # Montagem do grafo da DAG. A leitura das chamadas Python define as
    # dependências entre tasks (TaskFlow infere a partir do data flow):
    raw = extract()
    curated = curate(raw)
    # As duas tasks abaixo recebem o MESMO input (`curated`) e são independentes
    # entre si => o Airflow as executa EM PARALELO, formando dois branches do
    # DAG após `curate`. Isso reduz o tempo total da DAG quando há slots livres.
    build_features_tabular(curated)
    build_features_text(curated)


pipeline_ingestao()
