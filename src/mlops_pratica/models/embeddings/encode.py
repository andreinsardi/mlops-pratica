"""Codificação de textos em vetores densos com sentence-transformers.

Cada execução é logada no MLflow como um run do experimento de embeddings,
registrando: nome do modelo, dimensionalidade, quantidade de itens vetorizados,
amostra de inputs/outputs (artifact).

Sobre `sentence-transformers`
-----------------------------
Biblioteca da UKP-TUDarmstadt construída sobre HuggingFace Transformers.
Especializada em transformar TEXTOS INTEIROS em VETORES DENSOS (sentence
embeddings) — não em prever próxima palavra.

Modelo usado: `all-MiniLM-L6-v2`
- 6 camadas de transformer, ~22M parâmetros.
- Output: vetores de 384 dimensões.
- Treinado com aprendizado contrastivo em >1 bilhão de pares de sentenças
  -> capta similaridade SEMÂNTICA, não só lexical.
- Multilingual razoável (treinado em inglês mas funciona em pt-br para
  muitos casos; para qualidade superior em pt, usar `paraphrase-multilingual-MiniLM`).
- Trade-off: ~10x mais rápido que `all-mpnet-base-v2` com ~90% da qualidade.
  Ideal para demonstração em sala e para a maioria dos casos de uso.

`normalize_embeddings=True`
---------------------------
Normaliza cada vetor para norma L2 = 1 (vetor unitário). Por que importa?
- Com vetores unitários, o produto interno (dot product) é IGUAL ao
  cosseno: a · b = |a||b|cos(θ) = 1 · 1 · cos(θ) = cos(θ).
- Isso permite usar `<#>` (dot product) no pgvector como atalho mais
  rápido que `<=>` (cosseno), em alguns índices.
- Mais importante: garante que a distância no banco e a distância na
  query usem a MESMA escala — qualquer assimetria some.

Cache do modelo (lazy + global)
-------------------------------
Carregar um modelo HuggingFace custa segundos (download + parse + warm-up).
Em serving (FastAPI) faríamos isso UMA vez no startup; em jobs Airflow,
queremos carregar UMA vez por processo e reusar entre batches. O cache
de módulo (`_model_cache`) faz exatamente isso, sem precisar de Singleton
formal.

Logging no MLflow
-----------------
Embeddings não têm "métrica de acurácia". Mas faz sentido criar um run
de MLflow para CADA execução porque queremos rastrear:
- Qual modelo foi usado (versão do encoder afeta resultados);
- Quantos textos foram processados;
- Amostras de input para auditoria.
Esse experimento fica SEPARADO do experimento do classificador — eles
têm ciclos de vida diferentes.
"""

from __future__ import annotations

import logging
from typing import Iterable

import mlflow
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from mlops_pratica.config import settings

logger = logging.getLogger(__name__)

# Cache de módulo: vive enquanto o processo Python estiver vivo.
# Não é thread-safe em sentido estrito, mas o `SentenceTransformer` em si
# faz inferência segura entre threads, e a primeira chamada concorrente,
# no pior caso, recria a instância — não vaza nem corrompe.
_model_cache: SentenceTransformer | None = None


def get_encoder() -> SentenceTransformer:
    """Lazy-load do modelo. Reusa instância dentro do processo.

    Primeira chamada: baixa pesos do HuggingFace Hub (~80MB) e mantém
    em memória. Chamadas seguintes: retorno O(1).
    """
    global _model_cache
    if _model_cache is None:
        logger.info("Carregando encoder %s", settings.embedding_model_name)
        _model_cache = SentenceTransformer(settings.embedding_model_name)
    return _model_cache


def encode_texts(texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
    """Encoda uma lista de textos em uma matriz (N, dim).

    Parâmetros do `.encode`:
    - `batch_size=32`: tamanho do batch passado ao modelo. Valor maior
      acelera (mais paralelismo na GPU/CPU) mas consome mais memória.
    - `show_progress_bar=False`: sem barra; logs MLflow já dão observabilidade.
    - `convert_to_numpy=True`: devolve np.ndarray em vez de tensor torch
      -> mais fácil de manipular fora do contexto de treino.
    - `normalize_embeddings=True`: vetores unitários (veja docstring do módulo).
    """
    model = get_encoder()
    vecs = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs


def encode_dataframe(df: pd.DataFrame, text_col: str = "title") -> tuple[pd.DataFrame, str]:
    """Adiciona coluna `embedding` ao dataframe e loga no MLflow.

    Retorna (df_com_embedding, run_id).

    O `run_id` retornado é PROPAGADO para o pgvector (coluna `model_run_id`)
    -> permite rastrear, dado um vetor no banco, EXATAMENTE qual execução
    do encoder o produziu (auditoria/reprocessamento seletivo).
    """
    # Aponta para o tracking server e seleciona o experimento DE EMBEDDINGS
    # (diferente do experimento do classificador).
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_embeddings)

    df = df.copy()

    with mlflow.start_run(run_name="encode_titles") as run:
        mlflow.set_tag("modelo", settings.embedding_model_name)
        # Params: configurações fixas da execução.
        mlflow.log_param("embedding_model", settings.embedding_model_name)
        mlflow.log_param("embedding_dim", settings.embedding_dim)
        mlflow.log_param("n_inputs", len(df))

        vecs = encode_texts(df[text_col].tolist())
        # `list(vecs)` quebra a matriz em lista de 1 vetor por linha, para
        # virar coluna de DataFrame (cada célula = np.array de dim 384).
        df["embedding"] = list(vecs)

        # Amostras como artifact JSON: sanity check rápido na UI do MLflow.
        # Se a inferência der estranho, dá pra inspecionar os textos.
        sample = df[["id", text_col]].head(10).to_dict(orient="records")
        mlflow.log_dict({"sample_inputs": sample}, "samples.json")
        # Métricas numéricas: aparecem em gráficos da UI.
        mlflow.log_metric("n_vectors", len(vecs))
        mlflow.log_metric("vector_dim", vecs.shape[1])

        logger.info("Encode concluído: %d vetores (run_id=%s)", len(vecs), run.info.run_id)
        return df, run.info.run_id
