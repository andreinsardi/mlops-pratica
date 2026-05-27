"""DAG 3: Treino diário de embeddings + indexação em pgvector.

Esta DAG transforma os títulos das stories em vetores densos (embeddings)
usando um modelo SentenceTransformer e os indexa numa tabela Postgres
equipada com a extensão `pgvector`. O resultado habilita busca semântica
("encontre stories similares a esta frase") na API de serving.

Fluxo (3 tasks em série):
  1) load_text  -> lê features_text.parquet (produzido pela DAG 1)
  2) encode     -> gera embeddings + loga o run no MLflow Tracking
  3) index      -> faz upsert das linhas (id, vetor, metadados) no pgvector

Por que rodar às 03:00 (1h depois da DAG 2)?
  - Separa a carga computacional: DAG 2 ocupa CPU/memória para treino
    sklearn/XGBoost; DAG 3 ocupa para inferência do encoder + I/O de DB.
  - Evita contention sobre o MLflow Tracking server (escritas concorrentes
    de runs em paralelo aumentam latência).
  - Sequenciamento explícito por horário é mais simples e debugável que
    usar `ExternalTaskSensor` para encadear as duas DAGs.

Modelo escolhido: `all-MiniLM-L6-v2` (sentence-transformers).
  - 384 dimensões: leve, rápido, baixo footprint de memória/storage.
  - Padrão de mercado para prototipagem semântica em inglês.
  - Para produção em PT-BR, considerar `multilingual-e5-base` ou
    `paraphrase-multilingual-MiniLM-L12-v2`.

pgvector + HNSW:
  - `pgvector` é uma extensão do Postgres que adiciona tipos VECTOR e
    operadores de distância (<->, <#>, <=>) para busca por similaridade.
  - O índice HNSW (Hierarchical Navigable Small World) acelera consultas
    de "k-NN aproximado" em milhões de vetores — escolha de facto para
    serving de embeddings sem stack dedicada (Pinecone, Weaviate, etc.).
"""

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
    # Cron "0 3 * * *" = todo dia às 03:00 (1 hora depois da DAG de treino
    # preditivo, para escalonar carga sobre MLflow e o worker do Airflow).
    schedule="0 3 * * *",
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
        """Valida que existem textos para processar.

        Gate análogo ao `MIN_TRAIN_SAMPLES` da DAG 2, porém mais simples:
        aqui exigimos apenas que o parquet não esteja vazio. Embeddings de
        zero linhas não fazem sentido e gerariam falha confusa no encoder.
        """
        df = read_parquet(features_text_path())
        if df.empty:
            raise ValueError("features_text vazio.")
        return {"path": features_text_path(), "n_rows": len(df)}

    @task
    def encode(info: dict) -> dict:
        """Codifica os títulos em vetores e registra o run no MLflow.

        `encode_dataframe` cuida de:
          - carregar o SentenceTransformer (com cache in-process: ver nota
            abaixo sobre `_model_cache`);
          - rodar `model.encode(...)` em batch sobre a coluna `title`;
          - abrir um run no MLflow Tracking, logar parâmetros (nome do
            modelo, dimensão, número de linhas) e retornar o `run_id` para
            rastreabilidade.

        Sobre o cache do encoder: o módulo `models/embeddings/encode.py`
        mantém um `_model_cache` (dict no escopo do processo). Na primeira
        chamada, o SentenceTransformer é baixado/carregado em memória
        (pode levar segundos). Chamadas subsequentes NO MESMO PROCESSO
        são essencialmente gratuitas. Isso explica por que recodificar
        na próxima task tem custo baixo se o worker for o mesmo.
        """
        df = read_parquet(info["path"])
        df_emb, run_id = encode_dataframe(df, text_col="title")
        # IMPORTANTE (decisão didática): não devolvemos o DataFrame com os
        # vetores (que pode ter MBs de arrays float32) via XCom. Em vez disso,
        # passamos apenas o caminho do parquet original + o run_id; a task
        # `index` recodifica em memória aproveitando o cache do encoder.
        #
        # Em produção, o padrão correto seria salvar `df_emb` num bucket
        # intermediário (ex.: s3://staging/embeddings/<run_id>.parquet) e
        # passar adiante apenas esse caminho — assim a recodificação é
        # eliminada e a task `index` fica totalmente isolada do encoder.
        return {"path": info["path"], "run_id": run_id}

    @task
    def index(info: dict) -> int:
        """Faz upsert dos embeddings na tabela pgvector e retorna a contagem.

        Recodifica os textos (barato por causa do cache do encoder) e chama
        `index_embeddings`, que executa um INSERT ... ON CONFLICT (id) DO
        UPDATE no Postgres — upsert idempotente. Stories já indexadas têm
        seu vetor atualizado, novas são inseridas.

        O `run_id` é guardado junto a cada linha para rastrear qual run do
        MLflow gerou aquele embedding (linhagem entre dado servido e modelo).
        """
        df = read_parquet(info["path"])
        df_emb, _ = encode_dataframe(df, text_col="title")
        return index_embeddings(df_emb, run_id=info["run_id"])

    # Grafo linear: load_text -> encode -> index.
    info = load_text()
    enc = encode(info)
    index(enc)


pipeline_treino_embeddings()
