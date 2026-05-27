"""Preparação de features textuais (títulos) para o pipe de embeddings.

Diferente do `tabular.py`, aqui NÃO calculamos números — só limpamos e
padronizamos o input que vai para o encoder de embeddings
(`sentence-transformers`). O modelo de embeddings cuida da tokenização,
normalização de unicode, lowercasing etc. internamente.

A nossa responsabilidade é garantir:
- Só processar "story" (não "job"/"poll" — eles têm semântica diferente);
- Título não-vazio (encoder devolve vetor zero para string vazia, o que
  bagunça similaridade cosseno);
- Sem duplicatas por id (cada post entra UMA vez no índice vetorial);
- Metadados auxiliares (url, score, autor, timestamp) preservados para
  serem retornados na busca.
"""

from __future__ import annotations

import pandas as pd


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna dataframe com colunas mínimas para encoder de embeddings.

    Saída: id, title, url, score, by_author, created_ts
    """
    df = df.copy()
    # Filtra só stories (tem título como conteúdo principal).
    df = df[df["type"].fillna("") == "story"]
    # Sem título ou sem tempo, descarta — não dá pra encodar nem datar.
    df = df.dropna(subset=["title", "time"])
    # `.str.strip()` remove espaços/quebras de linha nas pontas — caso
    # comum de "string que parece existir mas é só whitespace".
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 0]

    # Tempo em UTC (mesma convenção do tabular.py); metadados normalizados.
    df["created_ts"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df["by_author"] = df["by"].fillna("unknown")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["url"] = df["url"].fillna("")

    keep = ["id", "title", "url", "score", "by_author", "created_ts"]
    # `drop_duplicates(subset=["id"])` garante 1 embedding por post — o
    # upsert no pgvector já protege contra isso, mas economiza compute.
    return df[keep].drop_duplicates(subset=["id"]).reset_index(drop=True)
