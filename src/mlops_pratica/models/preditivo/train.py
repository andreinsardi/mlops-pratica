"""Treino do classificador binário 'vai_bombar' com sklearn + XGBoost.

Compara dois modelos em runs separados do MLflow e registra o vencedor
no Model Registry (transição para Staging se AUC > baseline).
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

NUMERIC_FEATS = ["title_len", "n_words_title", "hour", "weekday", "has_url", "has_question"]
CATEGORICAL_FEATS = ["domain", "by_author"]


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False),
                CATEGORICAL_FEATS,
            ),
        ]
    )


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
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
    """Treina um modelo e loga no MLflow. Retorna (run_id, métricas)."""
    pipe = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor()),
            ("clf", estimator),
        ]
    )

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("modelo", name)
        mlflow.log_params({f"clf__{k}": v for k, v in estimator.get_params().items()
                           if isinstance(v, (int, float, str, bool, type(None)))})

        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = _evaluate(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        signature = mlflow.models.infer_signature(X_test, y_proba)
        if isinstance(estimator, XGBClassifier):
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                signature=signature,
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
    """Pipeline completo: split, treina 2 modelos, registra o melhor por AUC."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_preditivo)

    X = df[feature_columns()]
    y = df[target_column()].values

    if len(np.unique(y)) < 2:
        raise ValueError("Target binário tem apenas uma classe — dataset insuficiente.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results: dict[str, dict[str, Any]] = {}

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    rf_run_id, rf_metrics = train_one("random_forest", rf, X_train, X_test, y_train, y_test,
                                      register_name=register_name)
    results["random_forest"] = {"run_id": rf_run_id, "metrics": rf_metrics}

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    xgb_run_id, xgb_metrics = train_one("xgboost", xgb, X_train, X_test, y_train, y_test,
                                        register_name=register_name)
    results["xgboost"] = {"run_id": xgb_run_id, "metrics": xgb_metrics}

    # vencedor por AUC
    winner = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    results["winner"] = winner
    logger.info("Vencedor: %s (AUC=%.4f)", winner, results[winner]["metrics"]["roc_auc"])

    return results
