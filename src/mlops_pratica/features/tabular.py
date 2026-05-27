"""Engenharia de features tabulares para classificador 'vai bombar' (score>=100).

Sobre feature engineering
-------------------------
Feature engineering é o processo de transformar dados crus em SINAIS que o
modelo consegue aprender. Aqui partimos de stories do HackerNews (campos
soltos: title, url, time, by, score, ...) e produzimos variáveis que
TEORICAMENTE explicam a popularidade do post:

- Tamanho do título: títulos muito longos ou muito curtos performam diferente.
- Hora/dia da semana: posts às 9h de terça pegam mais audiência que às 3h de
  domingo.
- Tem URL externa? Posts "Ask HN" sem URL têm dinâmica diferente.
- Termina com "?": pergunta tende a gerar engajamento (descendants).
- Domínio do link (github.com, nytimes.com, ...): proxy de credibilidade.
- Autor: usuários conhecidos da comunidade já chegam com vantagem.

A regra de ouro: features tabulares são apenas DICAS. O modelo decide
quanto cada uma pesa.

Sobre o target binário `vai_bombar`
-----------------------------------
Score é um inteiro contínuo (pode ir de 1 a milhares). Poderíamos tratar
como regressão, mas:
- Para o caso de uso ("este post vai bombar?"), a decisão é binária.
- Classificação binária permite métricas mais interpretáveis (ROC AUC, PR AUC).
- Threshold de 100 é um corte didático razoável: na prática do HN, atingir
  100 upvotes já é "front page" e separa bem stories medianas de virais.

Encoding das categóricas
------------------------
As colunas `domain` e `by_author` são STRINGS de alta cardinalidade.
Aqui NÃO codificamos elas — devolvemos as strings cruas. O OneHotEncoder
do `train.py` (dentro do ColumnTransformer) é quem faz isso, com a opção
`min_frequency` para agrupar categorias raras em "infrequent_sklearn".
Isso é importante: encoding feito FORA do Pipeline gera data leakage,
porque o ajuste vê todo o dataset (treino + teste).
"""

from __future__ import annotations

from urllib.parse import urlparse

import numpy as np
import pandas as pd

# Threshold do alvo binário: score >= 100 => vai_bombar = 1.
# Valor escolhido para ser didático com poucos dados; ajustar para
# produção real (analisar a distribuição e equilibrar as classes).
SCORE_THRESHOLD = 100


def _extract_domain(url: object) -> str:
    """Extrai o domínio (netloc) de uma URL, com fallback robusto.

    Usado também no serving para garantir que a feature de domínio
    seja construída EXATAMENTE da mesma forma no treino e na inferência
    (evita "training-serving skew").
    """
    if not isinstance(url, str) or not url:
        return "no_url"
    try:
        # `urlparse(...).netloc` devolve "sub.dominio.com:porta".
        # Padronizamos para lowercase e removemos a porta.
        netloc = urlparse(url).netloc.lower()
        return netloc.split(":")[0] or "no_url"
    except Exception:
        return "no_url"


def build_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe stories curadas e devolve dataset pronto para treino.

    Saída tem as colunas:
      id, title_len, n_words_title, hour, weekday, has_url, has_question,
      domain, by_author, type, score, vai_bombar
    """
    df = df.copy()

    # Limpeza: só "story" (descarta job, ask, poll); precisa de título, tempo e score.
    df = df[df["type"].fillna("") == "story"]
    df = df.dropna(subset=["title", "time", "score"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["title"] = df["title"].astype(str)
    # `time` da API HN vem como Unix epoch em segundos. Convertemos para datetime
    # com `utc=True` — é boa prática manter tudo em UTC internamente e converter
    # para timezone local só na apresentação.
    df["created_ts"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["created_ts"])

    # ------------------ Features derivadas do título ------------------
    df["title_len"] = df["title"].str.len()             # tamanho em caracteres
    df["n_words_title"] = df["title"].str.split().str.len()  # contagem de palavras
    # ------------------ Features temporais (sazonalidade) -------------
    df["hour"] = df["created_ts"].dt.hour          # 0-23
    df["weekday"] = df["created_ts"].dt.weekday    # 0=segunda, 6=domingo
    # ------------------ Flags binárias --------------------------------
    df["has_url"] = df["url"].fillna("").astype(str).str.len().gt(0).astype(int)
    # `\?$` = pergunta no FINAL da string. Posts tipo "How do I...?" tendem
    # a gerar discussão (Ask HN).
    df["has_question"] = df["title"].str.contains(r"\?$", regex=True).astype(int)
    # ------------------ Categóricas (vão para OneHotEncoder no Pipeline) ----
    df["domain"] = df["url"].apply(_extract_domain)
    # `unknown` como sentinel: o OneHotEncoder vai tratar consistentemente.
    df["by_author"] = df["by"].fillna("unknown")

    # ------------------ Target binário --------------------------------
    df["vai_bombar"] = (df["score"] >= SCORE_THRESHOLD).astype(int)

    keep = [
        "id",
        "title",
        "title_len",
        "n_words_title",
        "hour",
        "weekday",
        "has_url",
        "has_question",
        "domain",
        "by_author",
        "type",
        "url",
        "score",
        "created_ts",
        "vai_bombar",
    ]
    return df[keep].reset_index(drop=True)


def feature_columns() -> list[str]:
    """Colunas (X) usadas no treino do modelo preditivo.

    Centralizar isso aqui evita divergência: o treino, a validação e o
    serving DEVEM enxergar exatamente as mesmas colunas, na mesma ordem.
    """
    return [
        "title_len",
        "n_words_title",
        "hour",
        "weekday",
        "has_url",
        "has_question",
        "domain",
        "by_author",
    ]


def target_column() -> str:
    """Nome da coluna alvo (y)."""
    return "vai_bombar"
