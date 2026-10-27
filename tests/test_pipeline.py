"""Testes unitários do núcleo de ML (sem HTTP)."""

import numpy as np

from app.ml.features import RAW_FEATURES, TARGET, engenharia_features, gerar_dataset
from app.ml.model import calcular_metricas, expected_calibration_error, faixa_risco, ks_statistic, prob_para_score


def test_dataset_sintetico_reprodutivel():
    a, b = gerar_dataset(500, seed=7), gerar_dataset(500, seed=7)
    assert a.equals(b)
    assert set(RAW_FEATURES + [TARGET]) <= set(a.columns)
    assert 0.4 < a[TARGET].mean() < 0.9


def test_engenharia_features_sem_nan_nas_derivadas():
    df = gerar_dataset(300, seed=1).fillna({"renda_mensal": 3000})
    eng = engenharia_features(df)
    for c in ["parcela_estimada", "comprometimento_renda", "lti", "score_por_consulta"]:
        assert eng[c].notna().all() and (eng[c] >= 0).all()


def test_score_monotonico_e_faixas():
    scores = [prob_para_score(p) for p in np.linspace(0.01, 0.99, 50)]
    assert scores == sorted(scores)
    assert 0 <= min(scores) and max(scores) <= 1000
    assert faixa_risco(900)[0] == "AA" and faixa_risco(100)[0] == "D"


def test_metricas_classificador_perfeito():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    m = calcular_metricas(y, p)
    assert m["roc_auc"] == 1.0 and m["ks"] == 1.0 and m["gini"] == 1.0
    assert ks_statistic(y, p) == 1.0
    assert expected_calibration_error(y, p) < 0.05
