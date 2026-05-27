"""Configurações centrais lidas de variáveis de ambiente.

Toda a stack roda em Docker; as variáveis são definidas em docker-compose.yml
e .env. Para uso local (testes), há valores default sensatos.

Por que centralizar em um objeto Settings?
-------------------------------------------
- Single source of truth: TODOS os módulos importam `settings` daqui. Se
  amanhã trocarmos o nome de uma variável (ex: PG_HOST -> POSTGRES_HOST),
  mudamos em UM lugar só.
- Tipagem: ler `os.getenv("PG_PORT")` espalhado pelo código devolve string;
  esquecer um `int(...)` causa bug silencioso (ex: comparações erradas).
  Aqui fazemos a conversão uma única vez.
- Testabilidade: como é uma `@dataclass(frozen=True)`, dá pra construir
  um `Settings(...)` alternativo em testes sem mexer no ambiente.
- Imutabilidade (`frozen=True`): impede que algum módulo "esperto" sobrescreva
  uma configuração em runtime e deixe um bug intermitente.

Padrão `os.getenv("VAR", default)`:
- Em Docker, `VAR` vem do docker-compose / .env.
- Fora do Docker (rodando testes locais ou notebooks), os defaults permitem
  importar este módulo sem precisar configurar nada.

Variáveis agrupadas por responsabilidade:
- MLflow (tracking + nomes de experimento);
- MinIO / S3 (object storage usado pelo MLflow para artefatos e pelas
  camadas raw/curated/features dos parquets);
- Postgres com extensão pgvector (armazena embeddings e permite busca
  vetorial usando o operador `<=>` para distância cosseno);
- Parâmetros da fonte de dados (HackerNews) e do modelo de embeddings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Container imutável de configuração. Instanciado uma vez no final do módulo."""


    # ---------------------------------------------------------------- MLflow
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow_experiment_preditivo: str = os.getenv(
        "MLFLOW_EXPERIMENT_PREDITIVO", "hn_classifier"
    )
    mlflow_experiment_embeddings: str = os.getenv(
        "MLFLOW_EXPERIMENT_EMBEDDINGS", "hn_embeddings"
    )

    # ---------------------------------------------------------------- MinIO/S3
    # MinIO é um object storage compatível com a API do S3 da AWS. Como ele
    # "fala" S3, podemos usar as MESMAS bibliotecas (s3fs, boto3) tanto em
    # dev (MinIO local) quanto em prod (AWS S3) — basta trocar o endpoint.
    # As três variáveis AWS_* abaixo são as credenciais que o cliente S3
    # consome por padrão; aqui apontam para o MinIO via `endpoint_url`.
    s3_endpoint_url: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    # Três buckets formam o "data lake" do projeto (padrão medallion simplificado):
    #   raw      -> dados crus exatamente como vieram da API HackerNews
    #   curated  -> stories limpas e deduplicadas
    #   features -> tabelas prontas para treino / encoder
    bucket_raw: str = "raw"
    bucket_curated: str = "curated"
    bucket_features: str = "features"

    # ---------------------------------------------------------------- Postgres app (pgvector)
    # `pgvector` é uma extensão do Postgres que adiciona o tipo `vector` e
    # operadores de distância (`<->` L2, `<=>` cosseno, `<#>` dot product).
    # Usamos esse Postgres para guardar os embeddings dos títulos e fazer
    # busca semântica direto no SQL — sem precisar de FAISS/Qdrant/etc.
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

    # ---------------------------------------------------------------- Treino
    # Threshold mínimo de exemplos para tentar treinar.
    # Default baixo para uso didático (demo em sala). Em produção, suba para
    # algo que dê estabilidade estatística (ex: 1000+).
    min_train_samples: int = int(os.getenv("MIN_TRAIN_SAMPLES", "20"))

    @property
    def pg_uri(self) -> str:
        """Constrói a URI no formato esperado pelo SQLAlchemy.

        Sintaxe: `postgresql+psycopg2://USER:PASS@HOST:PORT/DB`.
        O sufixo `+psycopg2` diz ao SQLAlchemy qual driver Python usar
        para conversar com o Postgres (existem outros como asyncpg).
        """
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


# Instância única importada por todo o projeto: `from mlops_pratica.config import settings`.
# Como `Settings` é frozen, esta instância é segura para compartilhar entre threads/módulos.
settings = Settings()
