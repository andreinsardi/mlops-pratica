"""Testes do cliente HN (mock de HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mlops_pratica.ingestion.hn_client import HackerNewsClient


def test_top_story_ids_limita_resultado():
    client = HackerNewsClient()
    with patch.object(client.session, "get") as mocked_get:
        resp = MagicMock()
        resp.json.return_value = list(range(500))
        resp.raise_for_status.return_value = None
        mocked_get.return_value = resp
        ids = client.top_story_ids(limit=10)
    assert ids == list(range(10))


def test_get_item_retorna_dict():
    client = HackerNewsClient()
    with patch.object(client.session, "get") as mocked_get:
        resp = MagicMock()
        resp.json.return_value = {"id": 1, "type": "story", "title": "x"}
        resp.raise_for_status.return_value = None
        mocked_get.return_value = resp
        item = client.get_item(1)
    assert item is not None
    assert item["id"] == 1


def test_get_item_retorna_none_se_removido():
    client = HackerNewsClient()
    with patch.object(client.session, "get") as mocked_get:
        resp = MagicMock()
        resp.json.return_value = None
        resp.raise_for_status.return_value = None
        mocked_get.return_value = resp
        item = client.get_item(999)
    assert item is None
