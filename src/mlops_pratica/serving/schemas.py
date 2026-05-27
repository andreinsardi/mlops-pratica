"""Pydantic schemas para FastAPI.

Por que Pydantic + FastAPI?
---------------------------
Pydantic é uma biblioteca de validação de dados baseada em type hints.
Quando uma rota do FastAPI declara `def predict(req: PredictRequest)`,
o framework AUTOMATICAMENTE:

1. Parseia o JSON do request body para um `PredictRequest`.
2. Valida cada campo:
   - tipo (int? str? float?);
   - obrigatoriedade (`...` significa "obrigatório", default significa "opcional");
   - restrições do `Field` (`ge=0, le=23` => greater-or-equal 0, less-or-equal 23).
3. Se falhar, retorna 422 Unprocessable Entity com erro detalhado APONTANDO
   exatamente qual campo e por quê — sem você escrever uma linha.
4. Gera a especificação OpenAPI das rotas, que vira a UI interativa em
   `/docs` (Swagger) e `/redoc` — documentação que NUNCA fica desatualizada
   porque é o próprio schema.

Cada classe abaixo é um CONTRATO: define o que entra e o que sai da API.
Manter os schemas em arquivo separado das rotas:
- Facilita reuso (cliente Python pode importar o mesmo schema);
- Permite ferramentas externas (openapi-generator) gerarem SDKs;
- Mantém o `app.py` focado em lógica.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Payload de entrada do POST /predict."""

    # `...` (Ellipsis) significa "campo obrigatório, sem default".
    # `description` aparece na UI do /docs como documentação ao vivo.
    title: str = Field(..., description="Título do post HN")
    # `Optional[str]` + default `None` = campo opcional no JSON.
    url: Optional[str] = Field(None, description="URL externa (opcional)")
    # Default "unknown" combina com o sentinel usado no treino (features/tabular.py).
    by_author: str = Field("unknown", description="Autor")
    # `ge=0, le=23` valida intervalo de hora. Cliente que mandar 25 leva
    # 422 ANTES de chegar na função de predict.
    hour: int = Field(..., ge=0, le=23)
    # Mesma ideia para dia da semana (0=segunda ... 6=domingo).
    weekday: int = Field(..., ge=0, le=6)


class PredictResponse(BaseModel):
    """Payload de saída do POST /predict.

    Declarar a resposta como tipo Pydantic é boa prática:
    FastAPI valida que a função realmente DEVOLVE o formato prometido,
    e a UI do /docs mostra um exemplo do output.
    """

    probability_vai_bombar: float
    prediction: int
    model_name: str
    model_stage: str


class SearchRequest(BaseModel):
    """Payload de entrada do POST /search."""

    # `min_length=1` evita query vazia (encoder devolveria vetor degenerado).
    query: str = Field(..., min_length=1)
    # Limite superior em `k` para proteger o banco de scans gigantes.
    k: int = Field(10, ge=1, le=50)


class SearchHit(BaseModel):
    """Um item da lista de resultados da busca semântica."""

    id: int
    title: str
    url: Optional[str]
    score: Optional[int]
    by_author: Optional[str]
    similarity: float


class SearchResponse(BaseModel):
    """Payload de saída do POST /search: query original + lista de hits."""

    query: str
    hits: list[SearchHit]
