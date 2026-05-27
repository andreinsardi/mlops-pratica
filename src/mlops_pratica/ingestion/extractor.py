"""Pipeline de extração: lista top stories e busca detalhes em paralelo.

A API HN exige UMA chamada por item (não há endpoint de batch). Para 100
stories, são 1 + 100 chamadas HTTP. Em série, com ~150ms de latência cada,
isso passa de 15 segundos. Com threads, baixa para 1-2 segundos.
"""

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

    Paralelismo
    -----------
    Usamos `ThreadPoolExecutor` porque o gargalo é I/O (esperar resposta
    HTTP), NÃO CPU. Durante a espera de rede a GIL do Python é liberada,
    então threads conseguem progredir de verdade — não é só ilusão.

    Por que threads e não asyncio?
    - `requests` é uma lib síncrona; misturar com asyncio exigiria
      `httpx`/`aiohttp` e reescrever o `HackerNewsClient` inteiro.
    - Para ~100 requests, threads são simples, suficientes e o código
      fica linear ("normal"). Asyncio só vale a pena na casa dos
      milhares de requests concorrentes.
    - `max_workers=16` é um meio-termo: paralelismo o bastante para
      esconder latência, mas sem sobrecarregar nem ser bloqueado por
      rate-limit da API.
    """
    client = client or HackerNewsClient()
    # Primeiro snapshot: lista de IDs. Esta é UMA chamada serial (rápida).
    ids = client.top_story_ids(limit=limit)

    items: list[dict[str, Any]] = []
    # O context manager garante shutdown limpo do pool mesmo se der exceção.
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Dispara todas as N requisições "de uma vez" — o pool agenda
        # `max_workers` por vez. Guardamos o mapeamento future->id para
        # logar QUAL id falhou se algo der errado.
        futures = {ex.submit(client.get_item, _id): _id for _id in ids}
        # `as_completed` itera conforme cada future fica pronto (ordem
        # de CHEGADA, não de submissão). Isso permite começar a processar
        # respostas rápidas sem esperar a mais lenta.
        for fut in as_completed(futures):
            item_id = futures[fut]
            try:
                item = fut.result()
            except Exception as exc:  # noqa: BLE001
                # Política: log e segue. Uma falha pontual num item não deve
                # derrubar a ingestão inteira. O retry interno do cliente HN
                # já tentou 3 vezes antes de chegar aqui.
                logger.warning("Falha ao buscar id=%s: %s", item_id, exc)
                continue
            if item is None:
                # Item deletado/inexistente — pula silenciosamente.
                continue
            # Carimbo de quando NÓS ingerimos (útil para auditoria e para
            # debug de drift de dados — diferente do `time` original do post).
            item["ingested_at"] = datetime.now(timezone.utc).isoformat()
            items.append(item)

    logger.info("Extração concluída: %d itens", len(items))
    return items
