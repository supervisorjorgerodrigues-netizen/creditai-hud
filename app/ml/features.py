"""
Definição de features, geração de dataset sintético realista e engenharia de atributos.

As funções deste módulo ficam em nível de módulo para serem serializáveis via joblib
(o FunctionTransformer do pipeline referencia `engenharia_features`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "aprovado"  # 1 = bom pagador / crédito aprovado

NUMERIC_RAW = [
    "idade",
    "pontuacao_credito",
    "renda_mensal",
    "valor_emprestimo",
    "prazo_meses",
    "tempo_emprego_anos",
    "inadimplencia_previa",
    "consultas_6m",
    "utilizacao_cartao",
]
CATEGORICAL = ["tipo_moradia", "finalidade_emprestimo"]
ENGINEERED = ["parcela_estimada", "comprometimento_renda", "lti", "score_por_consulta"]

RAW_FEATURES = NUMERIC_RAW + CATEGORICAL
NUMERIC_ALL = NUMERIC_RAW + ENGINEERED

MORADIAS = ["PROPRIA", "ALUGADA", "FINANCIADA", "OUTROS"]
FINALIDADES = ["EDUCACAO", "SAUDE", "NEGOCIOS", "PESSOAL", "QUITACAO_DIVIDAS", "IMOVEL"]

# Restrições monotônicas (+1 aumenta prob. de aprovação, -1 diminui, 0 livre).
# Garante coerência regulatória: score maior NUNCA pode piorar a decisão.
MONOTONIC = {
    "pontuacao_credito": 1,
    "renda_mensal": 1,
    "tempo_emprego_anos": 1,
    "inadimplencia_previa": -1,
    "consultas_6m": -1,
    "utilizacao_cartao": -1,
    "comprometimento_renda": -1,
    "lti": -1,
}

TAXA_MENSAL_REF = 0.025  # taxa de juros de referência para estimar a parcela (Price)


def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos financeiros derivados (usado DENTRO do pipeline — sem leakage)."""
    out = df.copy()
    i = TAXA_MENSAL_REF
    n = out["prazo_meses"].clip(lower=1).astype(float)
    fator = (i * (1 + i) ** n) / ((1 + i) ** n - 1)
    out["parcela_estimada"] = out["valor_emprestimo"] * fator
    renda = out["renda_mensal"].clip(lower=1.0)
    out["comprometimento_renda"] = out["parcela_estimada"] / renda
    out["lti"] = out["valor_emprestimo"] / (renda * 12.0)
    out["score_por_consulta"] = out["pontuacao_credito"] / (1.0 + out["consultas_6m"])
    return out


def gerar_dataset(n: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """Gera base sintética com relações causais plausíveis de risco de crédito."""
    rng = np.random.default_rng(seed)

    idade = rng.integers(18, 76, n)
    tempo_emprego = np.clip(rng.gamma(2.0, 2.8, n), 0, None)
    tempo_emprego = np.minimum(tempo_emprego, np.maximum(idade - 16, 0)).round(1)
    renda = np.round(np.exp(rng.normal(8.35, 0.62, n)) * (1 + 0.012 * (idade - 18)), 2)
    score = np.clip(rng.normal(640, 95, n) + 3.2 * tempo_emprego, 300, 850).astype(int)
    valor = np.round(np.clip(renda * rng.uniform(0.6, 9.0, n), 500, 500_000), 2)
    prazo = rng.choice([6, 12, 18, 24, 36, 48, 60], n, p=[0.06, 0.18, 0.12, 0.22, 0.2, 0.12, 0.1])
    inad = rng.binomial(1, np.clip(0.38 - (score - 300) / 1100, 0.02, 0.6))
    consultas = rng.poisson(np.clip(1.4 + inad * 2.2 - (score - 600) / 250, 0.2, 6))
    util = np.clip(rng.beta(2.2, 3.0, n) + inad * 0.18, 0, 1).round(3)
    moradia = rng.choice(MORADIAS, n, p=[0.38, 0.32, 0.22, 0.08])
    finalidade = rng.choice(FINALIDADES, n, p=[0.12, 0.08, 0.2, 0.3, 0.18, 0.12])

    df = pd.DataFrame(
        {
            "idade": idade,
            "pontuacao_credito": score,
            "renda_mensal": renda,
            "valor_emprestimo": valor,
            "prazo_meses": prazo,
            "tempo_emprego_anos": tempo_emprego,
            "inadimplencia_previa": inad,
            "consultas_6m": consultas,
            "utilizacao_cartao": util,
            "tipo_moradia": moradia,
            "finalidade_emprestimo": finalidade,
        }
    )
    eng = engenharia_features(df)

    logit = (
        1.1
        + 0.0105 * (df["pontuacao_credito"] - 640)
        + 0.55 * np.log(df["renda_mensal"] / 4000)
        + 0.06 * np.minimum(df["tempo_emprego_anos"], 15)
        - 1.35 * df["inadimplencia_previa"]
        - 0.22 * df["consultas_6m"]
        - 1.25 * (df["utilizacao_cartao"] - 0.4)
        - 3.4 * np.clip(eng["comprometimento_renda"] - 0.30, 0, None)
        - 0.25 * np.clip(eng["lti"] - 1.0, 0, None)
        + 0.012 * np.clip(df["idade"] - 25, -7, 25)
        + df["tipo_moradia"].map({"PROPRIA": 0.35, "FINANCIADA": 0.12, "ALUGADA": -0.1, "OUTROS": -0.25})
        + df["finalidade_emprestimo"].map(
            {"IMOVEL": 0.3, "EDUCACAO": 0.2, "NEGOCIOS": 0.05, "SAUDE": 0.0, "PESSOAL": -0.15, "QUITACAO_DIVIDAS": -0.45}
        )
    )
    logit = logit + rng.normal(0, 0.40, n)  # ruído idiossincrático
    prob = 1 / (1 + np.exp(-logit))
    df[TARGET] = rng.binomial(1, prob)

    # Injeta ~2% de valores ausentes para exercitar a imputação do pipeline
    for col in ["renda_mensal", "tempo_emprego_anos", "utilizacao_cartao"]:
        mask = rng.random(n) < 0.02
        df.loc[mask, col] = np.nan
    return df
