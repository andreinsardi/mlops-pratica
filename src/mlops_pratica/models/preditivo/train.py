"""Treino do classificador binário 'vai_bombar' com sklearn + XGBoost.

Compara dois modelos em runs separados do MLflow e registra o vencedor
no Model Registry (transição para Staging se AUC > baseline).

Conceitos centrais deste módulo
================================

1. sklearn Pipeline
-------------------
Um `Pipeline` ENCAPSULA pré-processamento + estimador como UMA unidade.
Por que importa? Porque:
- Quando chamamos `pipe.fit(X_train, y_train)`, o preprocessor (scaler,
  one-hot, etc.) aprende SÓ no treino — depois aplica os mesmos
  parâmetros no teste e no serving. Sem Pipeline, é trivial vazar
  estatísticas do teste para o treino (data leakage).
- Quando salvamos com `mlflow.sklearn.log_model(pipe, ...)`, salvamos o
  preprocessor JUNTO. No serving, basta `model.predict(novos_dados)` —
  não precisa reconstruir o preprocessor à mão (e correr risco de erro).

2. ColumnTransformer
--------------------
Aplica transformações DIFERENTES em colunas DIFERENTES dentro do mesmo
preprocessor. Aqui:
- Numéricas -> StandardScaler (média 0, desvio 1). Importante para modelos
  que dependem de escala (regressão logística, SVM, redes). Random Forest
  e XGBoost são invariantes a escala, mas escalar não atrapalha e mantém
  o código uniforme.
- Categóricas -> OneHotEncoder com `min_frequency=5`: categorias que
  aparecem menos de 5 vezes são agrupadas em "infrequent_sklearn".
  Por quê? Sem isso, autores que aparecem 1 vez viram uma coluna inteira
  só pra eles — o modelo decora ("overfit") e a coluna desaparece em
  produção.
- `handle_unknown="ignore"`: em serving, se aparecer um autor/domínio
  novo, o encoder gera vetor de zeros em vez de quebrar.

3. Random Forest vs XGBoost
---------------------------
- Random Forest: ensemble de árvores independentes via bagging. Baseline
  forte, pouco hiperparâmetro pra tunar, robusto a outliers.
- XGBoost: ensemble por boosting (cada árvore corrige o erro da anterior).
  Geralmente performa melhor que RF em dados tabulares, mas é mais
  sensível a hiperparâmetros.
Treinar OS DOIS e comparar é a abordagem "champion-challenger" — você
sempre tem um baseline para saber se o modelo novo está realmente melhor.

4. Métricas e quando usar cada uma
----------------------------------
- accuracy: % de acertos. SÓ é útil em dataset balanceado. Em "vai_bombar"
  geralmente <10% dos posts bombam, então um modelo bobo que sempre prevê
  "não bomba" tira ~90% de accuracy e é INÚTIL.
- F1: média harmônica entre precision e recall. Bom resumo único quando
  você se importa com a classe positiva e o dataset é desbalanceado.
- ROC AUC: área sob a curva ROC. Mede a capacidade do modelo de RANKEAR
  positivos acima de negativos, independente do threshold. Mais usada,
  mas pode ser otimista demais em dados muito desbalanceados.
- PR AUC (average_precision): área sob a curva precision-recall. MELHOR
  que ROC AUC quando a classe positiva é rara — foca em onde a ação
  acontece (lado dos positivos previstos).
Aqui escolhemos o vencedor por ROC AUC por simplicidade didática. Em
projeto real, escolher pela métrica que reflete o CUSTO de negócio.

5. train_test_split estratificado
---------------------------------
`stratify=y` garante que a PROPORÇÃO de positivos seja igual no treino
e no teste. Sem isso, com poucos dados, o teste pode pegar zero
positivos e quebrar as métricas que dependem da classe minoritária.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from mlops_pratica.config import settings
from mlops_pratica.features.tabular import (
    feature_columns,
    target_column,
)

logger = logging.getLogger(__name__)

# Separação por TIPO de feature: o ColumnTransformer aplica scaler nas numéricas
# e one-hot nas categóricas. Manter como constantes do módulo deixa explícito.
NUMERIC_FEATS = ["title_len", "n_words_title", "hour", "weekday", "has_url", "has_question"]
CATEGORICAL_FEATS = ["domain", "by_author"]


def _build_preprocessor() -> ColumnTransformer:
    """Monta o preprocessor que será o PRIMEIRO passo do Pipeline.

    Devolver um objeto NOVO a cada chamada é proposital: cada modelo
    treinado tem o SEU preprocessor (caso queiramos usar transformações
    diferentes por modelo no futuro, basta variar aqui).
    """
    return ColumnTransformer(
        transformers=[
            # Numéricas: padronização (z-score). Reversível e estabiliza
            # otimizadores que assumem features em escalas comparáveis.
            ("num", StandardScaler(), NUMERIC_FEATS),
            (
                "cat",
                # `min_frequency=5`: categorias com menos de 5 ocorrências viram
                #   uma única coluna agregada -> evita overfit em valores raros
                #   e mantém o número de colunas sob controle.
                # `handle_unknown="ignore"`: categoria nunca vista em treino
                #   (ex: novo autor) não quebra inference — vira vetor de zeros.
                # `sparse_output=False`: matriz densa (mais memória, mas
                #   compatível com mais estimadores; OK para datasets pequenos).
                OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False),
                CATEGORICAL_FEATS,
            ),
        ]
    )


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Calcula um conjunto comparável de métricas binárias.

    - `y_pred`: classes preditas (0/1) usando threshold de 0.5.
    - `y_proba`: probabilidades preditas para a classe positiva (necessário
      para ROC AUC e PR AUC, que NÃO dependem de threshold fixo).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # `zero_division=0`: se não houver positivos preditos, F1 ficaria
        # 0/0 e geraria warning -> forçamos para 0.0 silenciosamente.
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def train_one(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    register_name: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Treina UM modelo dentro de um run MLflow e retorna (run_id, métricas).

    Anatomia de um experimento MLflow:
    - `mlflow.start_run()` abre um RUN (1 experimento contém N runs).
      Cada chamada gera ID único, marcado com nome humano (`run_name`).
    - `set_tag` / `log_params` / `log_metrics`: anotações estruturadas.
      Tags = metadados qualitativos; params = hiperparâmetros; metrics =
      números monitoráveis em gráficos da UI.
    - `log_model` salva o pipeline serializado + assinatura + dependências.
      Se passar `registered_model_name`, o MLflow CRIA automaticamente
      uma versão nova no Model Registry — sem precisar de chamada extra.
    """
    # Pipeline = preprocessor + classificador como um único objeto treinável.
    # Esta é a estrutura que será serializada como artefato MLflow.
    pipe = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor()),
            ("clf", estimator),
        ]
    )

    with mlflow.start_run(run_name=name) as run:
        # Tags são úteis para filtrar runs na UI ("modelo=xgboost").
        mlflow.set_tag("modelo", name)
        # Logamos só hiperparâmetros do CLASSIFICADOR (com prefixo `clf__`
        # para combinar com a convenção do Pipeline). Filtramos para tipos
        # primitivos pra evitar erros de serialização do MLflow.
        mlflow.log_params({f"clf__{k}": v for k, v in estimator.get_params().items()
                           if isinstance(v, (int, float, str, bool, type(None)))})

        # FIT roda o ColumnTransformer e o estimador encadeados.
        pipe.fit(X_train, y_train)
        # `predict_proba` devolve P(classe_0) e P(classe_1). Pegamos a coluna
        # 1 porque é a probabilidade do "vai_bombar" (classe positiva).
        y_proba = pipe.predict_proba(X_test)[:, 1]
        # Threshold de 0.5 = padrão. Em produção, escolher por análise de
        # custo (ex: priorizar recall sobre precision -> baixar threshold).
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = _evaluate(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        # `infer_signature` examina os DTYPES de X e y para gerar um schema
        # (`ModelSignature`). Esse schema vira contrato no Model Registry:
        # consumidores conseguem ver "este modelo espera X colunas com Y
        # dtypes". Detecta mismatch em produção mais cedo.
        signature = mlflow.models.infer_signature(X_test, y_proba)
        # Observação didática: os dois ramos do if/else fazem a mesma coisa
        # (mlflow.sklearn.log_model). Manter o `isinstance` deixa explícito
        # que aqui poderíamos diferenciar (ex: mlflow.xgboost.log_model para
        # ter acesso ao formato nativo) — mas como o XGBClassifier está
        # DENTRO de um Pipeline sklearn, salvar como sklearn é correto.
        if isinstance(estimator, XGBClassifier):
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                signature=signature,
                # Se `register_name` for fornecido, MLflow registra uma nova
                # versão no Model Registry automaticamente neste mesmo passo.
                registered_model_name=register_name,
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                signature=signature,
                registered_model_name=register_name,
            )

        logger.info("Modelo %s: %s (run_id=%s)", name, metrics, run.info.run_id)
        return run.info.run_id, metrics


def train_and_register(df: pd.DataFrame, register_name: str | None = None) -> dict[str, Any]:
    """Pipeline completo: split, treina 2 modelos, registra o melhor por AUC.

    Fluxo:
    1. Configura MLflow (tracking URI + experimento ativo).
    2. Separa X (features) e y (alvo) usando as listas centralizadas em
       `features.tabular` — garante consistência treino/serving.
    3. Valida que existem AS DUAS classes (sem isso, sklearn quebra).
    4. Split estratificado 80/20.
    5. Treina e loga RF e XGB; cada um vira um run + uma versão no Registry.
    6. Elege o vencedor por ROC AUC.

    Observação: AMBOS os modelos viram versões no Registry. A promoção
    para Staging/Production fica a cargo do `register.py` (passo separado).
    """
    # Aponta o cliente MLflow para o tracking server (UI + backend).
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # `set_experiment` cria o experimento se não existir; runs subsequentes
    # caem nele automaticamente.
    mlflow.set_experiment(settings.mlflow_experiment_preditivo)

    X = df[feature_columns()]
    y = df[target_column()].values

    # Sanity check obrigatório: classificador binário precisa de duas classes.
    # Erro precoce e CLARO em vez de stack trace confuso lá no fit do sklearn.
    if len(np.unique(y)) < 2:
        raise ValueError("Target binário tem apenas uma classe — dataset insuficiente.")

    # `stratify=y` -> mesma proporção de classes no treino e no teste.
    # `random_state=42` -> reproducibilidade (mesma divisão em runs futuros).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results: dict[str, dict[str, Any]] = {}

    # ----- Modelo 1: Random Forest (baseline robusto, pouca tunagem) -----
    # n_estimators=200: ensemble de 200 árvores; mais árvores = mais estável
    # (até saturar). max_depth=12 limita complexidade individual (anti-overfit).
    # n_jobs=-1 = usa todos os cores disponíveis.
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    rf_run_id, rf_metrics = train_one("random_forest", rf, X_train, X_test, y_train, y_test,
                                      register_name=register_name)
    results["random_forest"] = {"run_id": rf_run_id, "metrics": rf_metrics}

    # ----- Modelo 2: XGBoost (gradient boosting, geralmente SOTA tabular) -----
    # Combinação clássica de hiperparâmetros: árvores rasas (max_depth=6),
    # learning_rate=0.1 (passo médio), subsample/colsample 0.9 introduzem
    # estocasticidade (anti-overfit, à la Random Forest). 300 árvores costuma
    # bastar para dataset desse tamanho.
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",  # objetivo binário padrão
        n_jobs=-1,
        random_state=42,
    )
    xgb_run_id, xgb_metrics = train_one("xgboost", xgb, X_train, X_test, y_train, y_test,
                                        register_name=register_name)
    results["xgboost"] = {"run_id": xgb_run_id, "metrics": xgb_metrics}

    # Vencedor por ROC AUC (escolha didática). Em produção: critério deve
    # refletir o custo de negócio (PR AUC para alvo raro, F1 com threshold
    # de operação, etc.) e ALÉM disso comparar com a versão em Production.
    winner = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    results["winner"] = winner
    logger.info("Vencedor: %s (AUC=%.4f)", winner, results[winner]["metrics"]["roc_auc"])

    return results
