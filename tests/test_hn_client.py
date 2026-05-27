"""Testes do cliente HN (mock de HTTP).

Estes testes ilustram um princípio fundamental: testes NÃO devem fazer
chamadas HTTP reais. Motivos:
  - Lentos: cada round-trip de rede adiciona dezenas/centenas de ms.
  - Frágeis: dependem do servidor de terceiros estar no ar e respondendo
    o que esperamos. Se a API HN mudar formato, TODOS os testes quebram.
  - Não determinísticos: top stories mudam minuto a minuto; um teste que
    afirma "primeiro id é 12345" vai falhar amanhã.

Solução: MOCK do `requests.Session.get` com `unittest.mock.patch.object`.
Substituímos o método em tempo de execução por um `MagicMock` que devolve
exatamente o que queremos. O cliente real nem percebe — chama `.get(...)`
normalmente, recebe nosso mock e segue o fluxo.

Padrão usado em todos os testes deste arquivo:
  1. Instancia o client (que cria a `requests.Session` real).
  2. `with patch.object(client.session, "get")`: substitui o método `get`
     da sessão DAQUELA instância pelo mock, só dentro do bloco `with`.
     Saindo do `with`, o método original volta automaticamente — zero
     vazamento entre testes.
  3. Configura o mock para devolver um response fake com .json() e
     .raise_for_status() controlados por nós.
  4. Chama o método do client e valida o resultado.

`MagicMock` é "esperto": qualquer atributo/método chamado nele é criado
automaticamente, devolvendo outro MagicMock. Usamos `return_value` para
fixar o que cada chamada devolve.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mlops_pratica.ingestion.hn_client import HackerNewsClient


def test_top_story_ids_limita_resultado():
    """Valida que o parâmetro `limit` recorta os IDs como esperado.

    A API HN devolve até ~500 IDs. Simulamos isso com `range(500)` e
    confirmamos que `limit=10` produz `[0..9]` (corte pelos primeiros).
    Test direto, dependência única: a função `top_story_ids` faz
    `[:limit]` no resultado.
    """
    client = HackerNewsClient()
    with patch.object(client.session, "get") as mocked_get:
        resp = MagicMock()
        resp.json.return_value = list(range(500))
        # raise_for_status precisa retornar None (não levantar) para
        # simular um 200 OK; sem isso, o cliente nem tentaria parsear.
        resp.raise_for_status.return_value = None
        mocked_get.return_value = resp
        ids = client.top_story_ids(limit=10)
    assert ids == list(range(10))


def test_get_item_retorna_dict():
    """Caminho feliz: item existente -> devolve o dict completo."""
    client = HackerNewsClient()
    with patch.object(client.session, "get") as mocked_get:
        resp = MagicMock()
        resp.json.return_value = {"id": 1, "type": "story", "title": "x"}
        resp.raise_for_status.return_value = None
        mocked_get.return_value = resp
        item = client.get_item(1)
    assert item is not None
    assert item["id"] == 1


def test_get_item_retorna_none_se_removido():
    """Caso especial da API HN: itens deletados retornam JSON `null`.

    Testar este caminho protege contra regressão da lógica `return item
    if item else None`. Sem este teste, alguém poderia "simplificar" o
    código para `return item` e quebrar silenciosamente quando o HN
    devolver um item deletado em produção.
    """
    client = HackerNewsClient()
    with patch.object(client.session, "get") as mocked_get:
        resp = MagicMock()
        resp.json.return_value = None
        resp.raise_for_status.return_value = None
        mocked_get.return_value = resp
        item = client.get_item(999)
    assert item is None
