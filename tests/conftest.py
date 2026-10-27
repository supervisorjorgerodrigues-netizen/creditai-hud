import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Isola os artefatos dos testes (não sobrescreve models/ e data/ do projeto)
os.environ.setdefault("CREDITAI_HOME", tempfile.mkdtemp(prefix="creditai_test_"))
os.environ.setdefault("CREDITAI_PAUSA_VISUAL", "0")

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    # O `with` dispara o lifespan → treina (se necessário) e carrega o modelo em RAM
    with TestClient(app) as c:
        yield c


@pytest.fixture
def cliente_valido():
    return {
        "nome_cliente": "Carlos Eduardo Santos",
        "idade": 35,
        "pontuacao_credito": 720,
        "renda_mensal": 7500.0,
        "valor_emprestimo": 18000.0,
        "prazo_meses": 24,
        "tempo_emprego_anos": 5.0,
        "inadimplencia_previa": 0,
        "consultas_6m": 1,
        "utilizacao_cartao": 0.3,
        "tipo_moradia": "PROPRIA",
        "finalidade_emprestimo": "NEGOCIOS",
    }
