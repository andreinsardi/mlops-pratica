"""Smoke tests da configuração."""

from __future__ import annotations

from mlops_pratica.config import settings


def test_settings_defaults():
    assert settings.embedding_dim == 384
    assert settings.bucket_raw == "raw"
    assert settings.bucket_curated == "curated"
    assert settings.bucket_features == "features"


def test_pg_uri_formata():
    uri = settings.pg_uri
    assert uri.startswith("postgresql+psycopg2://")
    assert settings.pg_user in uri
    assert settings.pg_db in uri
