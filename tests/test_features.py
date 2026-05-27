"""Testes de feature engineering.

Cobrem `build_tabular_features` e `build_text_features`. Note como cada
teste recebe `sample_hn_items: pd.DataFrame` como parâmetro: o pytest
inspeciona o nome do parâmetro e injeta a fixture homônima declarada
em `conftest.py`. Isso é "dependency injection by name" — recurso
nativo do pytest, sem precisar de decorator extra.

Estilo dos testes: padrão "Arrange-Act-Assert" simplificado.
  - Arrange: a fixture já preparou o DataFrame de entrada.
  - Act:     chama a função sob teste.
  - Assert:  verifica que o output bate com o esperado.

Cada teste cobre UM comportamento específico (granularidade fina):
quando um teste quebra, a mensagem aponta exatamente qual regra falhou
— muito melhor que um teste mega que testa tudo e quando falha você
não sabe onde começar a olhar.
"""

from __future__ import annotations

import pandas as pd

from mlops_pratica.features.tabular import (
    build_tabular_features,
    feature_columns,
    target_column,
)
from mlops_pratica.features.text import build_text_features


def test_build_tabular_features_basico(sample_hn_items: pd.DataFrame):
    """Caso geral: filtragem por type, colunas esperadas e target binário.

    Três asserts independentes que validam:
      1. Filtro: 5 items - 1 'job' = 4 stories devem sobrar.
      2. Contrato de colunas: todas as features declaradas em
         `feature_columns()` + a target estão presentes.
      3. Lógica do target: stories com score >= 100 viram vai_bombar=1.
    """
    feats = build_tabular_features(sample_hn_items)
    # Apenas type=story devem permanecer (4 de 5)
    assert len(feats) == 4
    for col in feature_columns() + [target_column()]:
        assert col in feats.columns
    # vai_bombar: ids 1 (score=250) e 3 (score=120) viralizam
    bombaram = feats[feats["vai_bombar"] == 1]["id"].tolist()
    assert set(bombaram) == {1, 3}


def test_build_tabular_extrai_domain(sample_hn_items: pd.DataFrame):
    """Domínio é extraído da URL e usa 'no_url' como sentinel quando ausente.

    Este teste blinda a função `_extract_domain` (também usada no serving)
    contra mudanças que quebrariam a paridade treino-serving — bug clássico
    e DIFÍCIL de notar em produção sem teste explícito.
    """
    feats = build_tabular_features(sample_hn_items)
    domains = feats.set_index("id")["domain"].to_dict()
    assert domains[1] == "github.com"
    assert domains[3] == "no_url"


def test_build_tabular_marca_pergunta(sample_hn_items: pd.DataFrame):
    """Flag `has_question` só é 1 quando o título termina com '?'."""
    feats = build_tabular_features(sample_hn_items)
    has_q = feats.set_index("id")["has_question"].to_dict()
    assert has_q[2] == 1  # "Why Rust is great?"
    assert has_q[1] == 0


def test_build_text_features_filtra_story(sample_hn_items: pd.DataFrame):
    """`build_text_features` mantém só stories e não calcula embedding.

    Verifica também o CONTRATO de colunas mínimas: o encoder downstream
    precisa de id, title, score e by_author. Importante reforçar com teste
    porque alterações no schema causam erro silencioso na DAG 3.
    """
    feats = build_text_features(sample_hn_items)
    assert len(feats) == 4
    assert "embedding" not in feats.columns
    assert {"id", "title", "score", "by_author"}.issubset(feats.columns)
