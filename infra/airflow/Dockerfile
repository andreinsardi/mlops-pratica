# =============================================================================
# Airflow custom image - inclui libs MLOps + pré-download do modelo de embeddings
# =============================================================================
FROM apache/airflow:2.9.3-python3.11

USER root

# Dependências de SO para sklearn, xgboost, sentence-transformers
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pré-baixa o modelo de embeddings para evitar download em runtime
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

ENV PYTHONPATH=/opt/airflow/src
