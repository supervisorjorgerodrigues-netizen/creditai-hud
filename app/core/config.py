"""
Configuração central do CreditAI HUD (caminhos relativos + hiperparâmetros padrão).
Todas as variáveis podem ser sobrescritas por variáveis de ambiente CREDITAI_*.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
# CREDITAI_HOME permite isolar dados/artefatos (ex.: testes usam um diretório temporário)
WORK_DIR = Path(os.getenv("CREDITAI_HOME", BASE_DIR))
DATA_DIR = WORK_DIR / "data"
MODELS_DIR = WORK_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
STATIC_DIR = BASE_DIR / "static"

DATASET_PATH = DATA_DIR / "credito_dataset.csv"
MODEL_PATH = MODELS_DIR / "creditai_model.joblib"
METRICS_PATH = MODELS_DIR / "relatorio_metricas.json"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

MODEL_NAME = "CreditAI-HGB-v2"
MODEL_VERSION = "2.0.0"

for _d in (DATA_DIR, MODELS_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(name: str, default, cast):
    raw = os.getenv(f"CREDITAI_{name}")
    return cast(raw) if raw is not None else default


@dataclass
class TrainConfig:
    """Hiperparâmetros do motor de treino (editáveis pelo HUD em Configurações)."""

    n_amostras: int = field(default_factory=lambda: _env("N_AMOSTRAS", 12_000, int))
    max_iter: int = field(default_factory=lambda: _env("MAX_ITER", 400, int))
    passo_iter: int = field(default_factory=lambda: _env("PASSO_ITER", 10, int))
    learning_rate: float = field(default_factory=lambda: _env("LEARNING_RATE", 0.035, float))
    max_leaf_nodes: int = 15
    max_depth: int = 6
    l2_regularization: float = 0.5
    paciencia: int = 8            # nº de passos sem melhora de AUC antes do early stopping
    limiar_decisao: float = 0.50  # ponto de corte da política de crédito
    seed: int = 42
    regerar_dataset: bool = False
    pausa_visual_s: float = field(default_factory=lambda: _env("PAUSA_VISUAL", 0.12, float))

    def to_dict(self) -> dict:
        return asdict(self)
