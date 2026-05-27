"""I/O em Postgres + pgvector via SQLAlchemy.

Por que SQLAlchemy CORE (e não o ORM)?
--------------------------------------
Aqui usamos só `create_engine` + `text(...)` — ou seja, o SQLAlchemy Core,
que é uma camada fina sobre o driver. NÃO usamos o ORM (classes mapeadas,
Session, queries fluentes).

Justificativa:
- O domínio é simples: 1-2 tabelas, queries em SQL puro com sintaxe
  específica do Postgres (pgvector, ON CONFLICT). Modelar isso com ORM
  agregaria boilerplate sem benefício real.
- Como temos um tipo customizado (`vector`) e operadores especiais
  (`<=>` para cosseno), expressar em SQL é mais natural e legível do
  que mapear com `TypeDecorator` no ORM.
- Engenheiros que sabem SQL leem isso na hora; ORM esconde o que
  realmente está rolando no banco.

Sobre `pgvector`
----------------
Extensão Postgres que adiciona:
- Tipo `vector(N)`: array float32 de tamanho fixo, armazenado de forma
  compacta no Postgres.
- Operadores de distância: `<->` (L2/Euclidiana), `<=>` (cosseno),
  `<#>` (dot product NEGATIVO — quanto MENOR, mais similar).
- Índices `ivfflat` / `hnsw` para busca aproximada (ANN) em milhões de
  vetores. Sem índice, a busca é exata (k-NN linear), o que é OK até
  ~100k vetores.

Como guardamos um vetor numpy no banco?
Não dá pra passar um np.ndarray como parâmetro SQL diretamente — o
psycopg2 não sabe convertê-lo. A convenção do pgvector é serializar
como string `'[0.1, 0.2, ...]'` e usar `CAST(... AS vector)` no SQL.
É o que `_vector_literal` faz.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from mlops_pratica.config import settings

logger = logging.getLogger(__name__)


def get_engine() -> Engine:
    """Cria um Engine SQLAlchemy reutilizável.

    Parâmetros importantes:
    - `pool_pre_ping=True`: antes de cada checkout, envia um SELECT 1 para
      garantir que a conexão ainda está viva. Crucial quando há firewall,
      load balancer ou Postgres que derruba conexões ociosas — sem isso,
      a primeira query depois de minutos parados quebra com "connection
      closed".
    - `future=True`: usa a API moderna 2.x do SQLAlchemy (semântica
      explícita de transação com `engine.begin()`).
    """
    return create_engine(settings.pg_uri, pool_pre_ping=True, future=True)


def _vector_literal(vec: Iterable[float]) -> str:
    """Serializa um vetor como literal pgvector: '[0.1,0.2,...]'.

    O pgvector aceita esse formato como input quando aplicamos
    `CAST(... AS vector)`. Usamos 6 casas decimais — suficiente para
    embeddings normalizados (que vivem em [-1, 1]) e mantém a string
    razoavelmente curta.
    """
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def upsert_embeddings(df: pd.DataFrame, model_name: str, run_id: str) -> int:
    """Upsert em batch na tabela `embeddings`.

    Espera colunas: id, title, url, score, by_author, created_ts, embedding (np.ndarray).

    Por que UPSERT (`ON CONFLICT DO UPDATE`)?
    -----------------------------------------
    Este pipeline roda periodicamente. Posts JÁ ingeridos antes podem
    aparecer de novo (com score atualizado). Queremos:
    - Inserir os novos (id ainda não existe);
    - Atualizar os existentes (score mudou, etc.);
    - Tudo em UMA única operação atômica.

    `INSERT ... ON CONFLICT (id) DO UPDATE SET ...` é a sintaxe nativa
    do Postgres para isso (também conhecida como "upsert"). `EXCLUDED.x`
    é uma pseudo-tabela que representa a linha que TENTOU ser inserida,
    permitindo copiar valores do INSERT para o UPDATE.

    Sem isso, teríamos que: SELECT pra ver se existe; INSERT ou UPDATE
    condicionalmente; lidar com race conditions. Tudo isso some com
    uma única query.
    """
    if df.empty:
        logger.info("Nada para upsert em embeddings.")
        return 0

    eng = get_engine()
    # `text(...)` cria uma "query SQL parametrizada". Os `:nome` viram
    # placeholders que o psycopg2 substitui com escaping correto — protege
    # contra SQL injection e converte tipos Python automaticamente.
    sql = text("""
        INSERT INTO embeddings
            (id, title, url, score, by_author, created_ts,
             embedding, model_name, model_run_id)
        VALUES
            (:id, :title, :url, :score, :by_author, :created_ts,
             CAST(:embedding AS vector), :model_name, :model_run_id)
        ON CONFLICT (id) DO UPDATE SET
            title        = EXCLUDED.title,
            url          = EXCLUDED.url,
            score        = EXCLUDED.score,
            by_author    = EXCLUDED.by_author,
            created_ts   = EXCLUDED.created_ts,
            embedding    = EXCLUDED.embedding,
            model_name   = EXCLUDED.model_name,
            model_run_id = EXCLUDED.model_run_id,
            indexed_at   = NOW();
    """)

    rows = []
    for r in df.itertuples(index=False):
        emb = r.embedding
        if isinstance(emb, np.ndarray):
            emb_lit = _vector_literal(emb.tolist())
        else:
            emb_lit = _vector_literal(emb)
        rows.append(
            {
                "id": int(r.id),
                "title": r.title,
                "url": r.url if pd.notna(r.url) else None,
                "score": int(r.score) if pd.notna(r.score) else None,
                "by_author": r.by_author if pd.notna(r.by_author) else None,
                "created_ts": r.created_ts,
                "embedding": emb_lit,
                "model_name": model_name,
                "model_run_id": run_id,
            }
        )

    # `engine.begin()` abre uma transação e commita/faz rollback automaticamente.
    # Passando uma LISTA de dicts para `execute`, o SQLAlchemy faz "executemany"
    # do driver — UM round-trip com todas as linhas em vez de N inserts.
    with eng.begin() as conn:
        conn.execute(sql, rows)
    logger.info("Upsert de %d embeddings concluído.", len(rows))
    return len(rows)


def search_similar(
    query_vec: np.ndarray, k: int = 10
) -> list[dict]:
    """Busca semântica por similaridade cosseno.

    Como funciona a query:
    - `embedding <=> CAST(:vec AS vector)` calcula a DISTÂNCIA cosseno
      (0 = idênticos, 1 = ortogonais, 2 = opostos). É um operador NATIVO
      do pgvector — quando há índice apropriado (ivfflat/hnsw), ele usa
      ANN; sem índice, faz scan linear (OK para datasets pequenos).
    - `1 - distancia` converte para SIMILARIDADE cosseno (1 = idênticos,
      0 = ortogonais). É só uma conveniência de apresentação — o ranking
      é o mesmo.
    - `ORDER BY embedding <=> ...` ordena do mais próximo ao mais distante;
      `LIMIT :k` corta nos top-k.

    Observação didática: o cálculo cosseno é feito DENTRO do banco. Não
    trazemos os 384 floats de cada linha para Python — só os k resultados.
    Esse é o ganho de usar pgvector em vez de carregar tudo em memória.
    """
    eng = get_engine()
    vec_lit = _vector_literal(query_vec.tolist())
    sql = text(f"""
        SELECT id, title, url, score, by_author, created_ts,
               1 - (embedding <=> CAST(:vec AS vector)) AS similarity
        FROM embeddings
        ORDER BY embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """)
    with eng.connect() as conn:
        result = conn.execute(sql, {"vec": vec_lit, "k": k})
        # `r._mapping` expõe a linha como dict-like; convertemos para dict
        # puro para facilitar serialização posterior (ex: JSON do FastAPI).
        rows = [dict(r._mapping) for r in result]
    return rows
