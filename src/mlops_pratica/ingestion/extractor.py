"""Pipeline de extração: lista top stories e busca detalhes em paralelo."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from mlops_pratica.ingestion.hn_client import HackerNewsClient

logger = logging.getLogger(__name__)


def fetch_top_stories(
    client: HackerNewsClient | None = None,
    limit: int = 100,
    max_workers: int = 16,
) -> list[dict[str, Any]]:
    """Retorna os `limit` top stories com detalhes hidratados.

    Cada item retornado tem os campos:
      id, type, by, time, title, url, score, descendants, kids, deleted, dead
    Adicionado: `ingested_at` (UTC ISO).
    """
    client = client or HackerNewsClient()
    ids = client.top_story_ids(limit=limit)

    items: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(client.get_item, _id): _id for _id in ids}
        for fut in as_completed(futures):
            item_id = futures[fut]
            try:
                item = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha ao buscar id=%s: %s", item_id, exc)
                continue
            if item is None:
                continue
            item["ingested_at"] = datetime.now(timezone.utc).isoformat()
            items.append(item)

    logger.info("Extração concluída: %d itens", len(items))
    return items
