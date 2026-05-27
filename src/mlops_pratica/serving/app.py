"""FastAPI - endpoints /predict e /search.

GET  /health   -> liveness
POST /predict  -> classifica HN story (vai_bombar)
POST /search   -> busca semântica em pgvector

Por que FastAPI?
----------------
- Tipagem forte via Pydantic = validação automática dos payloads (4xx
  claros para o cliente, sem mexer numa linha).
- Geração automática de OpenAPI (/docs e /redoc) — documentação viva
  da API sem manter doc à parte.
- ASGI = suporta async, mas também aceita funções síncronas sem fricção
  (nossas rotas são síncronas porque o gargalo é CPU/IO bloqueante de
  sklearn/pgvector — async não traria ganho aqui).
- Performance comparável a frameworks Go/Node em benchmarks reais.

Padrão `lifespan` (em vez de `@app.on_event`)
---------------------------------------------
`@app.on_event("startup")` / `@app.on_event("shutdown")` foram
DEPRECADOS. O padrão moderno é um async context manager passado em
`FastAPI(lifespan=...)`:
    código antes do yield = startup
    código depois do yield = shutdown
Aqui usamos isso para CARREGAR O MODELO uma única vez no startup —
caro demais pra fazer por request.

Estado global vs DI
-------------------
Usamos um dict module-level `state` para simplicidade. Em projetos
maiores, o padrão é usar `Depends` do FastAPI para injetar o modelo
nas rotas (mais testável). Para este escopo educacional, o dict
basta.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from mlops_pratica.config import settings
# Importamos `_extract_domain` do módulo de features para garantir que
# a feature `domain` no serving seja calculada EXATAMENTE como no treino.
# Esta é uma das técnicas mais simples para evitar "training-serving skew".
from mlops_pratica.features.tabular import _extract_domain
from mlops_pratica.models.embeddings.encode import encode_texts
from mlops_pratica.serving.schemas import (
    PredictRequest,
    PredictResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from mlops_pratica.storage.pg_io import search_similar
from mlops_pratica.tracking.mlflow_utils import load_production_model

logger = logging.getLogger(__name__)

# Estado global preenchido no lifespan/startup.
# Manter como dict (não como variável solta) facilita "limpar tudo" no shutdown
# e adicionar mais coisas (cache, métricas Prometheus, etc.) sem reestruturar.
state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Carrega modelo do Registry na inicialização.

    Se falhar (ex: MLflow ainda não acessível, primeiro deploy sem modelo
    treinado), NÃO derruba a API: marca `state["model"] = None` e a rota
    /predict tenta carregar tardiamente. Isso é fundamental para a
    EXPERIÊNCIA DIDÁTICA — a API sobe, o /health funciona, e o aluno
    consegue treinar o modelo DEPOIS sem precisar reiniciar nada.
    """
    try:
        state["model"] = load_production_model(
            settings.model_name_registry, settings.model_stage
        )
        logger.info("Modelo %s/%s carregado", settings.model_name_registry, settings.model_stage)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Não foi possível carregar modelo no startup: %s", exc)
        state["model"] = None
    yield
    # Shutdown: limpa o estado (libera memória do modelo).
    state.clear()


app = FastAPI(
    title="MLOps Prática - HackerNews",
    description="API de predição (vai_bombar) e busca semântica (pgvector).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness/readiness simples.

    `model_loaded` permite ao orquestrador (Docker/Kubernetes) saber se a
    API está pronta para SERVIR. Aqui devolvemos 200 mesmo sem modelo
    (para a API subir sem dependência circular), mas em produção o ideal
    é separar `/health` (live) de `/ready` (ready) — o último só retorna
    200 quando o modelo está carregado.
    """
    return {
        "status": "ok",
        "model_loaded": state.get("model") is not None,
        "tracking_uri": settings.mlflow_tracking_uri,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Classifica uma HN story (vai_bombar) usando o modelo do Registry.

    A FastAPI já validou `req` contra `PredictRequest` (Pydantic) antes
    de chegar aqui — se o cliente mandou um hour=25, recebeu 422 sem
    nem entrar nesta função.
    """
    model = state.get("model")
    if model is None:
        # Lazy retry: se o startup falhou (ex: API subiu ANTES do MLflow
        # estar pronto), tenta carregar agora. Padrão útil em ambiente
        # de aula com docker-compose, onde a ordem de subida é frágil.
        try:
            state["model"] = load_production_model(
                settings.model_name_registry, settings.model_stage
            )
            model = state["model"]
        except Exception as exc:
            # 503 = Service Unavailable. Mensagem clara para o cliente
            # saber que é INFRA, não payload inválido.
            raise HTTPException(503, f"Modelo indisponível: {exc}") from exc

    # Construção da linha de input com EXATAMENTE as mesmas transformações
    # do `build_tabular_features` (em features/tabular.py). Como o Pipeline
    # carregado já contém o ColumnTransformer (StandardScaler + OneHotEncoder),
    # NÃO precisamos pré-processar nada aqui — basta entregar as colunas
    # cruas e o pipeline cuida do resto.
    row = {
        "title_len": len(req.title),
        "n_words_title": len(req.title.split()),
        "hour": req.hour,
        "weekday": req.weekday,
        "has_url": int(bool(req.url)),
        "has_question": int(req.title.rstrip().endswith("?")),
        # Reaproveita a MESMA função usada no treino -> zero risco de skew.
        "domain": _extract_domain(req.url),
        "by_author": req.by_author,
    }
    df = pd.DataFrame([row])
    # `predict_proba` devolve shape (1, 2): [P(não bomba), P(bomba)].
    proba = float(model.predict_proba(df)[0, 1])
    return PredictResponse(
        probability_vai_bombar=proba,
        prediction=int(proba >= 0.5),
        model_name=settings.model_name_registry,
        model_stage=settings.model_stage,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Busca semântica por similaridade cosseno em pgvector.

    Passos:
    1. Encodar a query do usuário usando o MESMO modelo de embeddings
       usado para indexar (consistência é tudo em busca vetorial).
    2. Delegar para `search_similar` (pg_io), que executa SQL com o
       operador `<=>` do pgvector e devolve os top-k resultados.
    3. Empacotar em Pydantic para serialização automática (e validação
       do que sai da função, não só do que entra).
    """
    # encode_texts devolve matriz (1, dim); pegamos o único vetor.
    vec = encode_texts([req.query])[0]
    rows = search_similar(vec, k=req.k)
    hits = [
        SearchHit(
            id=r["id"],
            title=r["title"],
            url=r["url"],
            score=r["score"],
            by_author=r["by_author"],
            similarity=float(r["similarity"]),
        )
        for r in rows
    ]
    return SearchResponse(query=req.query, hits=hits)
