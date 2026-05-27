"""I/O em MinIO via s3fs / pyarrow (parquet).

As camadas usadas (layout no estilo "Hive-partitioned"):
  - raw/hn/dt=YYYY-MM-DD/hr=HH/items.parquet
  - curated/hn/stories.parquet
  - features/tabular/train.parquet
  - features/text/titles.parquet

Bibliotecas em jogo
-------------------
- `s3fs`: expõe o S3 (e MinIO) como se fosse um filesystem POSIX. Com isso
  podemos usar `fs.open(...)` igualzinho ao `open(...)` do Python, e o
  pandas/pyarrow conseguem ler/escrever direto via objeto file-like. Isso
  evita ter que baixar o arquivo para disco temporariamente.
- `pyarrow`: implementação canônica do formato Parquet em Python. Parquet
  é colunar, comprimido e tipado — leitura seletiva de colunas é barata,
  o arquivo é ~5-10x menor que CSV e os tipos (datas, ints, floats) são
  preservados sem mágica de parsing. Padrão de fato em data engineering.

Por que esse layout particionado por dt/hr?
- Permite fazer "scan parcial" eficiente: para reprocessar só o dia X,
  basta ler `raw/hn/dt=X/`.
- Convenção compatível com Spark, Athena, DuckDB, Trino, etc. — se um
  dia migrarmos, o data lake "já está pronto".
- O nome `dt=YYYY-MM-DD/hr=HH` (chave=valor) é o estilo Hive, reconhecido
  automaticamente por essas engines como colunas de partição.

Por que três buckets (raw/curated/features)?
- É o padrão "medallion" simplificado (bronze/silver/gold):
  RAW = exatamente o que veio da API (auditável, reprocessável).
  CURATED = limpo, deduplicado, tipos corretos.
  FEATURES = pronto para alimentar treino/encoder (X já formatado).
- Separar permite reprocessar uma camada SEM ter que rebaixar dados da API.
"""

from __future__ import annotations

import logging
from io import BytesIO

import pandas as pd
import s3fs

from mlops_pratica.config import settings

logger = logging.getLogger(__name__)


def get_fs() -> s3fs.S3FileSystem:
    """Filesystem MinIO/S3 já autenticado.

    `client_kwargs={"endpoint_url": ...}` é a chave que faz o s3fs apontar
    para o MinIO em vez do S3 da AWS. Em produção AWS, basta omitir
    `endpoint_url` e usar credenciais reais — o resto do código não muda.
    """
    return s3fs.S3FileSystem(
        key=settings.aws_access_key_id,
        secret=settings.aws_secret_access_key,
        client_kwargs={"endpoint_url": settings.s3_endpoint_url},
    )


def write_parquet(df: pd.DataFrame, s3_path: str) -> None:
    """Grava DataFrame como parquet em MinIO. `s3_path` no formato `s3://bucket/key`."""
    fs = get_fs()
    # `fs.open(...)` devolve um file-like remoto; `to_parquet` escreve nele
    # como se fosse arquivo local. Sem `index=False`, o pandas gravaria o
    # índice como coluna extra "Unnamed: 0" — quase nunca o que queremos.
    with fs.open(s3_path, "wb") as f:
        df.to_parquet(f, index=False, engine="pyarrow")
    logger.info("Escrito %d linhas em %s", len(df), s3_path)


def read_parquet(s3_path: str) -> pd.DataFrame:
    """Lê parquet do MinIO."""
    fs = get_fs()
    # Lemos para memória primeiro (BytesIO) porque alguns leitores parquet
    # esperam um buffer seekable e o stream do s3fs nem sempre cumpre isso.
    # Para arquivos pequenos (até centenas de MB) o custo é desprezível.
    with fs.open(s3_path, "rb") as f:
        buf = BytesIO(f.read())
    df = pd.read_parquet(buf, engine="pyarrow")
    logger.info("Lido %d linhas de %s", len(df), s3_path)
    return df


def list_paths(s3_prefix: str) -> list[str]:
    """Lista objetos sob um prefixo `s3://bucket/prefix/`.

    Útil para descobrir partições dinamicamente (ex: listar todos os
    `dt=` disponíveis para um reprocessamento incremental).
    """
    fs = get_fs()
    return [f"s3://{p}" for p in fs.ls(s3_prefix.replace("s3://", ""))]


# ---------------------------------------------------------------------------
# Helpers de path: centralizam a CONVENÇÃO de nomes. Se um dia mudarmos o
# layout (ex: passar a particionar curated também), só essas funções precisam
# ser ajustadas — o resto do código já chama elas em vez de hardcodar strings.
# ---------------------------------------------------------------------------

def raw_path(dt: str, hr: str) -> str:
    """Path da camada RAW para uma janela hora-a-hora (dt=YYYY-MM-DD, hr=HH)."""
    return f"s3://{settings.bucket_raw}/hn/dt={dt}/hr={hr}/items.parquet"


def curated_path() -> str:
    """Path único da camada CURATED (snapshot consolidado de stories)."""
    return f"s3://{settings.bucket_curated}/hn/stories.parquet"


def features_tabular_path() -> str:
    """Features tabulares prontas para o classificador `vai_bombar`."""
    return f"s3://{settings.bucket_features}/tabular/train.parquet"


def features_text_path() -> str:
    """Features textuais (títulos limpos) prontas para o encoder de embeddings."""
    return f"s3://{settings.bucket_features}/text/titles.parquet"
