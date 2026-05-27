-- =============================================================================
-- Inicialização do Postgres
-- Cria 3 databases: airflow (metadata), mlflow (backend store), app (pgvector)
-- =============================================================================

-- Database para Airflow
CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow OWNER airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

-- Database para MLflow backend store
CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE DATABASE mlflow OWNER mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Database de aplicação (com pgvector)
CREATE USER app WITH PASSWORD 'app';
CREATE DATABASE app OWNER app;
GRANT ALL PRIVILEGES ON DATABASE app TO app;

-- Habilita pgvector no DB de aplicação
\c app
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabela de embeddings (dim=384 para all-MiniLM-L6-v2)
CREATE TABLE IF NOT EXISTS embeddings (
    id           BIGINT PRIMARY KEY,
    title        TEXT NOT NULL,
    url          TEXT,
    score        INTEGER,
    by_author    TEXT,
    created_ts   TIMESTAMPTZ,
    embedding    vector(384) NOT NULL,
    model_name   TEXT NOT NULL,
    model_run_id TEXT,
    indexed_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Índice HNSW para similaridade cosseno
CREATE INDEX IF NOT EXISTS embeddings_vector_idx
    ON embeddings USING hnsw (embedding vector_cosine_ops);

-- Índice de pesquisa rápida por autor / data
CREATE INDEX IF NOT EXISTS embeddings_author_idx ON embeddings (by_author);
CREATE INDEX IF NOT EXISTS embeddings_ts_idx ON embeddings (created_ts DESC);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO app;
