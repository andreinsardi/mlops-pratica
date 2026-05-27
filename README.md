# MLOps Prática — HackerNews end-to-end (100% Docker)

[![CI](https://github.com/andreinsardi/mlops-pratica/actions/workflows/ci.yml/badge.svg)](https://github.com/andreinsardi/mlops-pratica/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

Projeto didático de MLOps full-stack: ingestão de API pública, data lake, dois pipes de ML (preditivo tabular + embeddings), tracking/registry com MLflow e serving via FastAPI. Tudo orquestrado pelo Airflow, rodando em containers Docker.

**Material da disciplina MLOps — MBA em Inteligência Artificial e Analytics Aplicadas a Negócios (FGV).**

Autor: [André Insardi](https://github.com/andreinsardi) · ext.andre.insardi@prof.fgv.edu.br

## Stack

| Camada | Tecnologia |
|---|---|
| Orquestração | Apache Airflow 2.9 (LocalExecutor) |
| Tracking + Model Registry | MLflow 2.16 |
| Data Lake | MinIO (S3-compatível) |
| Metadata + Vector Store | Postgres 16 + pgvector 0.7 |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| ML Preditivo | scikit-learn + XGBoost |
| Serving | FastAPI + Uvicorn |
| CI | GitHub Actions (pytest + ruff + docker build) |

## Arquitetura

```
   ┌─────────────────────────┐
   │     HackerNews API      │
   └────────────┬────────────┘
                │  (extract)
                ▼
   ┌──────────────────────────────────────┐
   │   Airflow DAGs                       │
   │   • pipeline_ingestao    @hourly     │
   │   • pipeline_treino_pred  02:00      │
   │   • pipeline_treino_emb   03:00      │
   └────┬───────────────────────┬─────────┘
        │                       │
        ▼                       ▼
  ┌──────────┐            ┌──────────────┐
  │  MinIO   │ <──artifact─│  MLflow      │
  │  Lake    │   store     │  Tracking+   │
  │ raw/curat│             │  Registry    │
  │ features │             └──────┬───────┘
  └────┬─────┘                    │
       │                          ▼
       │                   ┌──────────────┐
       │                   │  FastAPI     │
       │                   │ /predict     │
       │                   │ /search      │
       │                   └──────┬───────┘
       │                          │
       ▼                          ▼
  ┌──────────────────────────────────────┐
  │   Postgres                            │
  │   • DB airflow (metadata)             │
  │   • DB mlflow (backend store)         │
  │   • DB app  (pgvector)                │
  └───────────────────────────────────────┘
```

## Quick start

```bash
# 1) preparar .env
cp .env.example .env

# 2) subir a stack
make up

# 3) acompanhar logs
make logs

# 4) acessar UIs
#   Airflow:  http://localhost:8080   (admin/admin)
#   MLflow:   http://localhost:5000
#   MinIO:    http://localhost:9001   (minioadmin/minioadmin)
#   FastAPI:  http://localhost:8000/docs
```

Após a stack subir:

1. Abra o **Airflow** e despause as 3 DAGs (`pipeline_ingestao`, `pipeline_treino_preditivo`, `pipeline_treino_embeddings`).
2. Dispare manualmente `pipeline_ingestao` algumas vezes para acumular dados.
3. Dispare `pipeline_treino_preditivo` e `pipeline_treino_embeddings`.
4. Veja o experimento em **MLflow** e o modelo em **Models**.
5. Teste o FastAPI:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Show HN: Tool for X","url":"https://x.com","by_author":"alice","hour":14,"weekday":1}'

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"machine learning operations","k":5}'
```

## Estrutura

```
mlops-pratica/
├── docker-compose.yml
├── .env.example / .env
├── Makefile
├── pyproject.toml / requirements-dev.txt
├── infra/                 # Dockerfiles e init scripts
├── dags/                  # 3 DAGs Airflow
├── src/mlops_pratica/     # código Python modular
│   ├── ingestion/         # cliente HN, extractor
│   ├── storage/           # MinIO + Postgres
│   ├── features/          # tabular + text
│   ├── models/            # preditivo + embeddings
│   ├── tracking/          # utilitários MLflow
│   └── serving/           # FastAPI
├── tests/                 # pytest
└── .github/workflows/     # CI GitHub Actions
```

## Operação

- **Testes:** `make test`
- **Lint:** `make lint`
- **Rebuild imagens:** `make build`
- **Reset total (apaga volumes):** `make clean`

## Notas didáticas

- **Versionamento triplo:**
  - **Código:** Git
  - **Dados:** parquet particionado em MinIO (camadas raw/curated/features)
  - **Modelos:** MLflow Tracking (runs) + Model Registry (Staging/Production)
- **Reprodutibilidade:** todas as runs do MLflow capturam params, métricas, signature e o pipeline sklearn completo (preprocessor + estimator).
- **Promoção didática:** a DAG promove automaticamente a última versão para Staging. Em produção real, exigir gate humano.

## Deploy em GCP VM

Veja o guia `docs/deploy_gcp_vm.md` para subir essa stack em uma Compute Engine VM.

## Licença

Material educacional - uso livre.
