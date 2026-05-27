# Makefile - atalhos para o projeto MLOps Prática
.PHONY: help up down logs ps build restart clean test lint env

help:
	@echo "Targets disponíveis:"
	@echo "  make env        - cria .env a partir do .env.example"
	@echo "  make up         - sobe toda a stack (Postgres, MinIO, MLflow, Airflow, FastAPI)"
	@echo "  make down       - derruba a stack"
	@echo "  make logs       - tail dos logs de todos os serviços"
	@echo "  make ps         - status dos containers"
	@echo "  make build      - rebuild das imagens"
	@echo "  make restart    - down + up"
	@echo "  make clean      - down + remove volumes (CUIDADO: apaga dados)"
	@echo "  make test       - roda pytest local"
	@echo "  make lint       - roda ruff"
	@echo ""
	@echo "URLs:"
	@echo "  Airflow:  http://localhost:8080  (admin/admin)"
	@echo "  MLflow:   http://localhost:5000"
	@echo "  MinIO:    http://localhost:9001  (minioadmin/minioadmin)"
	@echo "  FastAPI:  http://localhost:8000/docs"

env:
	@test -f .env || cp .env.example .env
	@echo ".env pronto."

up: env
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

build:
	docker compose build --no-cache

restart: down up

clean:
	docker compose down -v
	@echo "Volumes removidos."

test:
	PYTHONPATH=src pytest -v

lint:
	ruff check src tests dags
