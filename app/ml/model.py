"""
Construção do pipeline, métricas de risco de crédito e o artefato calibrado servido pela API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from app.ml.features import (
    CATEGORICAL,
    MONOTONIC,
    NUMERIC_ALL,
    RAW_FEATURES,
    engenharia_features,
)


def construir_pipeline(cfg) -> Pipeline:
    """Pipeline sem leakage: engenharia → imputação/encoding → GBDT com restrições monotônicas."""
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_ALL),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    clf = HistGradientBoostingClassifier(
        learning_rate=cfg.learning_rate,
        max_iter=cfg.passo_iter,
        max_leaf_nodes=cfg.max_leaf_nodes,
        max_depth=cfg.max_depth,
        l2_regularization=cfg.l2_regularization,
        min_samples_leaf=40,
        monotonic_cst=MONOTONIC,
        early_stopping=False,  # early stopping é controlado pelo motor (checkpoint do pico)
        warm_start=True,       # treino incremental → telemetria ao vivo no HUD
        random_state=cfg.seed,
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(engenharia_features, validate=False)),
            ("pre", pre),
            ("clf", clf),
        ]
    )


# ----------------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------------
def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def calcular_metricas(y, p, limiar: float = 0.5, particao: str = "") -> dict[str, Any]:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    yhat = (p >= limiar).astype(int)
    auc = roc_auc_score(y, p)
    return {
        "particao": particao,
        "amostras": int(len(y)),
        "roc_auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y, p), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece": round(expected_calibration_error(y, p), 4),
        "acuracia": round(float(accuracy_score(y, yhat)), 4),
        "precisao": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "taxa_aprovacao": round(float(yhat.mean()), 4),
        "matriz_confusao": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def curva_calibracao(y, p, bins: int = 10) -> list[dict]:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            out.append({"previsto": round(float(p[m].mean()), 4), "observado": round(float(y[m].mean()), 4), "n": int(m.sum())})
    return out


# ----------------------------------------------------------------------------------
# Score 0–1000 (escala tipo bureau, método PDO de scorecards)
# ----------------------------------------------------------------------------------
PDO, BASE_SCORE, BASE_ODDS = 60.0, 600.0, 3.0


def prob_para_score(p: float) -> int:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    fator = PDO / math.log(2)
    offset = BASE_SCORE - fator * math.log(BASE_ODDS)
    return int(min(1000, max(0, round(offset + fator * math.log(p / (1 - p))))))


def faixa_risco(score: int) -> tuple[str, str]:
    if score >= 800:
        return "AA", "Risco Muito Baixo"
    if score >= 680:
        return "A", "Baixo Risco"
    if score >= 560:
        return "B", "Risco Moderado"
    if score >= 440:
        return "C", "Risco Elevado"
    return "D", "Alto Risco"


# ----------------------------------------------------------------------------------
# Calibração
# ----------------------------------------------------------------------------------
class PlattCalibrator:
    """
    Calibração sigmoide (Platt) sobre o logit da probabilidade bruta.
    Suave e monotônica — não "satura" em 0/1 como a isotônica em bases pequenas,
    o que mantém o score 0–1000 contínuo e estável nas extremidades.
    """

    def fit(self, p_raw, y) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        z = self._logit(p_raw).reshape(-1, 1)
        self.lr_ = LogisticRegression(C=1e3).fit(z, np.asarray(y))
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict(self, p_raw) -> np.ndarray:
        return self.lr_.predict_proba(self._logit(p_raw).reshape(-1, 1))[:, 1]


# ----------------------------------------------------------------------------------
# Artefato servido
# ----------------------------------------------------------------------------------
@dataclass
class CreditAIModel:
    """Pipeline GBDT + calibração Platt + perfil de referência para explicabilidade."""

    pipeline: Pipeline
    calibrador: PlattCalibrator | IsotonicRegression
    perfil_referencia: dict[str, Any]
    limiar: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def prob_bruta(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[RAW_FEATURES])[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.calibrador.predict(self.prob_bruta(df)), 1e-4, 1 - 1e-4)

    def explicar(self, linha: dict[str, Any]) -> tuple[float, list[dict]]:
        """
        Contribuição por atributo via substituição pelo perfil de referência
        (occlusion attribution): Δp = p(cliente) − p(cliente com atributo = referência).
        Um único batch → latência de poucos ms.
        """
        base = pd.DataFrame([linha])[RAW_FEATURES]
        variantes = [base]
        for col in RAW_FEATURES:
            v = base.copy()
            v[col] = self.perfil_referencia[col]
            variantes.append(v)
        lote = pd.concat(variantes, ignore_index=True)
        probs = self.predict_proba(lote)
        p0 = float(probs[0])
        contribs = [
            {"atributo": col, "valor": linha[col], "impacto": round(p0 - float(probs[i + 1]), 4)}
            for i, col in enumerate(RAW_FEATURES)
        ]
        contribs.sort(key=lambda c: abs(c["impacto"]), reverse=True)
        return p0, contribs
