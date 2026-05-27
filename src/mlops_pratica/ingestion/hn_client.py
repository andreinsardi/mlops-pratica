"""Cliente HTTP para a Firebase API do HackerNews.

Documentação oficial: https://github.com/HackerNews/API
Endpoints usados:
  - GET /v0/topstories.json  -> lista de IDs (até ~500)
  - GET /v0/item/{id}.json   -> detalhe do post
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
    """Cliente leve para a API pública do HackerNews."""

    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        self.base_url = (base_url or settings.hn_api_base).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _get(self, path: str) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def top_story_ids(self, limit: int | None = None) -> list[int]:
        """Retorna IDs das top stories (ordenadas por score corrente)."""
        ids = self._get("topstories.json") or []
        if limit is not None:
            ids = ids[:limit]
        logger.info("HackerNews: %d top story ids", len(ids))
        return ids

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        """Retorna detalhe de um item; None se removido/inexistente."""
        item = self._get(f"item/{item_id}.json")
        return item if item else None
