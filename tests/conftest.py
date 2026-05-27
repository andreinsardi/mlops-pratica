"""Fixtures compartilhadas dos testes.

O arquivo `conftest.py` é especial no pytest: tudo que for definido aqui
fica automaticamente DISPONÍVEL para todos os testes no mesmo diretório
(e subdiretórios), sem precisar importar explicitamente. É o local
canônico para fixtures, hooks e configurações compartilhadas.

Por que ter `sample_hn_items` como fixture em vez de constante?
- Pytest injeta a fixture na função de teste que a declara como
  parâmetro -> dependência explícita e localizada.
- Cada teste recebe uma INSTÂNCIA NOVA do DataFrame -> testes não
  podem se contaminar mutuamente (isolamento).
- Se um dia precisarmos parametrizar (ex: gerar 100 items aleatórios),
  basta trocar a implementação da fixture; nenhum teste muda.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Truque clássico: adiciona `src/` ao sys.path para que `import mlops_pratica.*`
# funcione mesmo sem ter rodado `pip install -e .` no ambiente de testes.
# `parents[1]` sobe um nível a partir de `tests/conftest.py` -> raiz do projeto.
# Útil em ambiente de aula onde o aluno pode rodar `pytest` direto sem instalar.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# `# noqa: E402` silencia a regra do flake8 que exige imports no topo do arquivo:
# aqui PRECISAMOS importar depois de manipular o sys.path, senão o import falha.
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def sample_hn_items() -> pd.DataFrame:
    """DataFrame mínimo representando 5 stories do HN.

    Dataset cuidadosamente construído para exercitar todos os caminhos
    importantes da feature engineering:
      - id=1: story com score alto (>=100) e URL do github  -> vai_bombar=1, domain=github.com
      - id=2: story com pergunta no título e score baixo    -> has_question=1, vai_bombar=0
      - id=3: Ask HN sem URL (score alto)                   -> domain=no_url, vai_bombar=1
      - id=4: story comum (score < 100)                     -> vai_bombar=0
      - id=5: type='job' (deve ser FILTRADO fora)           -> não aparece nas features

    Timestamps `1700000000+` são epochs Unix de novembro/2023, mantidos
    fixos para que features temporais (hour, weekday) sejam DETERMINÍSTICAS.
    """
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
