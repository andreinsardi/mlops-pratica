"""Cliente HTTP para a Firebase API do HackerNews.

Documentação oficial: https://github.com/HackerNews/API
Endpoints usados:
  - GET /v0/topstories.json  -> lista de IDs (até ~500)
  - GET /v0/item/{id}.json   -> detalhe do post

Por que este módulo existe?
---------------------------
A API do HN é pública, gratuita e idempotente (GETs sem efeitos colaterais),
mas é um servidor de terceiros: pode ter latência variável, devolver 5xx
esporádicos ou cortar conexões. Encapsular o acesso aqui nos dá:
- Um único ponto para configurar timeouts, retry e headers;
- Reuso de conexão TCP (via `requests.Session`), o que reduz drasticamente
  a latência quando buscamos centenas de itens em sequência;
- Resiliência declarativa via `tenacity` (decorador de retry com backoff
  exponencial) — sem `try/except` espalhados no resto do código.

Sobre idempotência: como TODOS os endpoints usados são GET puros, podemos
re-tentar sem medo de criar duplicatas ou efeitos colaterais. Essa
propriedade é o que JUSTIFICA o retry automático.
"""

from __future__ import annotations

import logging
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mlops_pratica.config import settings

logger = logging.getLogger(__name__)


class HackerNewsClient:
    """Cliente leve para a API pública do HackerNews.

    Encapsula uma `requests.Session` (pool de conexões keep-alive) e
    aplica retry com backoff exponencial via `tenacity` em todas as
    chamadas HTTP.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        self.base_url = (base_url or settings.hn_api_base).rstrip("/")
        self.timeout = timeout
        # `requests.Session` mantém um pool de conexões TCP abertas (keep-alive).
        # Se chamarmos `get_item` 100 vezes seguidas, NÃO faremos 100 handshakes
        # TCP+TLS — a sessão reaproveita a mesma conexão. Em pipelines de
        # ingestão isso costuma cortar o tempo total em mais da metade.
        self.session = requests.Session()

    @retry(
        # `tenacity` é uma biblioteca de retry declarativo. Em vez de escrever
        # `for attempt in range(3): try: ... except: sleep(2**attempt)`,
        # decoramos a função e configuramos a política aqui de forma explícita.
        stop=stop_after_attempt(3),  # no máximo 3 tentativas (1 original + 2 retries)
        # Backoff exponencial: espera 1s, 2s, 4s... até no máximo 10s entre
        # tentativas. Isso evita martelar um servidor que já está sofrendo
        # ("thundering herd") e dá tempo para um 503 transitório se recuperar.
        wait=wait_exponential(multiplier=1, min=1, max=10),
        # Só faz retry para erros DA REDE / DO HTTP. Bugs nossos (KeyError,
        # TypeError, etc.) sobem na hora — não queremos esconder bugs com retry.
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,  # se TODAS as tentativas falharem, propaga a exceção original
    )
    def _get(self, path: str) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, timeout=self.timeout)
        # `raise_for_status` transforma respostas 4xx/5xx em exceção, o que
        # ativa o retry acima quando for um 5xx (subclasse de RequestException).
        resp.raise_for_status()
        return resp.json()

    def top_story_ids(self, limit: int | None = None) -> list[int]:
        """Retorna IDs das top stories (ordenadas por score corrente).

        Esta lista MUDA em tempo real conforme posts ganham/perdem upvotes.
        Por isso fazemos o snapshot UMA vez e depois buscamos detalhes em
        paralelo (ver `ingestion/extractor.py`).
        """
        ids = self._get("topstories.json") or []
        if limit is not None:
            ids = ids[:limit]
        logger.info("HackerNews: %d top story ids", len(ids))
        return ids

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        """Retorna detalhe de um item; None se removido/inexistente.

        A API HN devolve `null` para itens deletados. Tratamos isso aqui
        para que o chamador receba simplesmente `None` em vez de um dict
        vazio ambíguo.
        """
        item = self._get(f"item/{item_id}.json")
        return item if item else None
