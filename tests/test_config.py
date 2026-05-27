"""Smoke tests da configuração.

"Smoke test" = teste rápido e raso que apenas confirma que o objeto/módulo
sobe sem explodir. Não cobre lógica complexa — serve como CANARIO: se
algum dia alguém quebrar o `Settings` (ex: renomear um atributo usado
em todo lugar), este teste falha em milissegundos e evita debug longo.

São o tipo de teste mais barato de manter e o mais rentável: rodam em
ms, raramente quebram à toa, e protegem contra regressões grosseiras.
"""

from __future__ import annotations

from mlops_pratica.config import settings


def test_settings_defaults():
    """Garante que os valores default críticos do `Settings` não mudaram.

    Esses valores são usados em todo o projeto (nomes de bucket nos paths
    do MinIO, dimensão do encoder na criação da tabela pgvector, etc.).
    Mudança acidental aqui = bug silencioso em produção.
    """
    assert settings.embedding_dim == 384
    assert settings.bucket_raw == "raw"
    assert settings.bucket_curated == "curated"
    assert settings.bucket_features == "features"


def test_pg_uri_formata():
    """Valida o formato da connection string Postgres montada pela property.

    Estratégia: não comparamos a string inteira (depende de host/user que
    podem mudar via env vars), checamos só os INVARIANTES:
      - prefixo do driver correto (psycopg2);
      - user e db efetivamente embutidos na URI.
    Esse padrão "asserts sobre propriedades estruturais" é robusto a
    mudanças razoáveis de configuração.
    """
    uri = settings.pg_uri
    assert uri.startswith("postgresql+psycopg2://")
    assert settings.pg_user in uri
    assert settings.pg_db in uri
