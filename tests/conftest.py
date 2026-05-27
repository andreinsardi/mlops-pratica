"""Fixtures compartilhadas dos testes."""

from __future__ import annotations

import sys
from pathlib import Path

# Garante import dos módulos sem instalar o pacote
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def sample_hn_items() -> pd.DataFrame:
    """DataFrame mínimo representando 5 stories do HN."""
    return pd.DataFrame(
        [
            {"id": 1, "type": "story", "by": "alice", "time": 1700000000,
             "title": "Show HN: Cool MLOps project", "url": "https://github.com/cool",
             "score": 250, "descendants": 30},
            {"id": 2, "type": "story", "by": "bob", "time": 1700003600,
             "title": "Why Rust is great?", "url": "https://blog.rust.org/post",
             "score": 5, "descendants": 1},
            {"id": 3, "type": "story", "by": "carol", "time": 1700007200,
             "title": "Ask HN: What is your favorite editor",
             "url": None, "score": 120, "descendants": 80},
            {"id": 4, "type": "story", "by": "alice", "time": 1700010800,
             "title": "OpenAI releases something new",
             "url": "https://openai.com/news", "score": 50, "descendants": 10},
            {"id": 5, "type": "job", "by": "company", "time": 1700014400,
             "title": "Hiring senior engineer", "url": "https://x.com/job",
             "score": 1, "descendants": 0},
        ]
    )
