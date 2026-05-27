# =============================================================================
# FastAPI Serving - /predict + /search
# =============================================================================
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pré-baixa o modelo de embeddings (para o endpoint /search)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "mlops_pratica.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
