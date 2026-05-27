"""I/O em MinIO via s3fs / pyarrow (parquet).

As camadas usadas:
  - raw/hn/dt=YYYY-MM-DD/hr=HH/items.parquet
  - curated/hn/stories.parquet
  - features/tabular/train.parquet
  - features/text/titles.parquet
"""

from __future__ import annotations

import logging
from io import BytesIO

import pandas as pd
import s3fs

from mlops_pratica.config import settings

logger = logging.getLogger(__name__)


def get_fs() -> s3fs.S3FileSystem:
    """Filesystem MinIO/S3 já autenticado."""
    return s3fs.S3FileSystem(
        key=settings.aws_access_key_id,
        secret=settings.aws_secret_access_key,
        client_kwargs={"endpoint_url": settings.s3_endpoint_url},
    )


def write_parquet(df: pd.DataFrame, s3_path: str) -> None:
    """Grava DataFrame como parquet em MinIO. `s3_path` no formato `s3://bucket/key`."""
    fs = get_fs()
    with fs.open(s3_path, "wb") as f:
        df.to_parquet(f, index=False, engine="pyarrow")
    logger.info("Escrito %d linhas em %s", len(df), s3_path)


def read_parquet(s3_path: str) -> pd.DataFrame:
    """Lê parquet do MinIO."""
    fs = get_fs()
    with fs.open(s3_path, "rb") as f:
        buf = BytesIO(f.read())
    df = pd.read_parquet(buf, engine="pyarrow")
    logger.info("Lido %d linhas de %s", len(df), s3_path)
    return df


def list_paths(s3_prefix: str) -> list[str]:
    """Lista objetos sob um prefixo `s3://bucket/prefix/`."""
    fs = get_fs()
    return [f"s3://{p}" for p in fs.ls(s3_prefix.replace("s3://", ""))]


def raw_path(dt: str, hr: str) -> str:
    return f"s3://{settings.bucket_raw}/hn/dt={dt}/hr={hr}/items.parquet"


def curated_path() -> str:
    return f"s3://{settings.bucket_curated}/hn/stories.parquet"


def features_tabular_path() -> str:
    return f"s3://{settings.bucket_features}/tabular/train.parquet"


def features_text_path() -> str:
    return f"s3://{settings.bucket_features}/text/titles.parquet"
