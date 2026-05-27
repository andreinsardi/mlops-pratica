"""Testes de feature engineering."""

from __future__ import annotations

import pandas as pd

from mlops_pratica.features.tabular import (
    build_tabular_features,
    feature_columns,
    target_column,
)
from mlops_pratica.features.text import build_text_features


def test_build_tabular_features_basico(sample_hn_items: pd.DataFrame):
    feats = build_tabular_features(sample_hn_items)
    # Apenas type=story devem permanecer (4 de 5)
    assert len(feats) == 4
    for col in feature_columns() + [target_column()]:
        assert col in feats.columns
    # vai_bombar: ids 1 (score=250) e 3 (score=120) viralizam
    bombaram = feats[feats["vai_bombar"] == 1]["id"].tolist()
    assert set(bombaram) == {1, 3}


def test_build_tabular_extrai_domain(sample_hn_items: pd.DataFrame):
    feats = build_tabular_features(sample_hn_items)
    domains = feats.set_index("id")["domain"].to_dict()
    assert domains[1] == "github.com"
    assert domains[3] == "no_url"


def test_build_tabular_marca_pergunta(sample_hn_items: pd.DataFrame):
    feats = build_tabular_features(sample_hn_items)
    has_q = feats.set_index("id")["has_question"].to_dict()
    assert has_q[2] == 1  # "Why Rust is great?"
    assert has_q[1] == 0


def test_build_text_features_filtra_story(sample_hn_items: pd.DataFrame):
    feats = build_text_features(sample_hn_items)
    assert len(feats) == 4
    assert "embedding" not in feats.columns
    assert {"id", "title", "score", "by_author"}.issubset(feats.columns)
