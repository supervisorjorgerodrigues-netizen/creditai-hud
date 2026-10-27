"""
Camada de serviço (Singleton) — inferência em RAM, regras de negócio e telemetria de latência.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import joblib
import numpy as np

from app.core.config import MODEL_PATH
from app.ml.features import TAXA_MENSAL_REF
from app.ml.model import CreditAIModel, faixa_risco, prob_para_score


class RegraNegocioError(ValueError):
    """Violação de regra de negócio financeira → HTTP 400."""


class ModeloIndisponivelError(RuntimeError):
    """Modelo ainda não carregado → HTTP 503."""


class CreditoService:
    def __init__(self) -> None:
        self.model_path = MODEL_PATH
        self.modelo: CreditAIModel | None = None
        self._lock = threading.RLock()
        self.latencias: deque[float] = deque(maxlen=500)
        self.total_predicoes = 0

    # ---------------------------------------------------------------- ciclo de vida
    def carregar_modelo_em_ram(self) -> None:
        with self._lock:
            self.modelo = joblib.load(self.model_path)

    def substituir_modelo(self, modelo: CreditAIModel) -> None:
        """Hot-swap atômico chamado pelo motor de treino."""
        with self._lock:
            self.modelo = modelo

    def descarregar_modelo_da_ram(self) -> None:
        with self._lock:
            self.modelo = None

    @property
    def carregado(self) -> bool:
        return self.modelo is not None

    def latencia_p50(self) -> float | None:
        return round(float(np.percentile(self.latencias, 50)), 3) if self.latencias else None

    # ---------------------------------------------------------------- regras
    @staticmethod
    def validar_regras_negocio(d: dict[str, Any]) -> None:
        idade_laboral = max(d["idade"] - 14, 0)
        if d["tempo_emprego_anos"] > idade_laboral:
            raise RegraNegocioError(
                f"Inconsistencia cadastral: tempo_emprego_anos ({d['tempo_emprego_anos']}) excede a "
                f"idade laboral possivel ({idade_laboral} anos)."
            )
        if d["idade"] + d["prazo_meses"] / 12 > 85:
            raise RegraNegocioError("Politica de credito: idade ao fim do contrato nao pode exceder 85 anos.")
        if d["valor_emprestimo"] > d["renda_mensal"] * 120:
            raise RegraNegocioError(
                "Politica de credito: valor solicitado excede 120x a renda mensal declarada."
            )

    # ---------------------------------------------------------------- inferência
    def prever(self, dados: dict[str, Any], limiar: float | None = None) -> dict[str, Any]:
        if self.modelo is None:
            raise ModeloIndisponivelError("Modelo não carregado em memória.")
        self.validar_regras_negocio(dados)

        t0 = time.perf_counter()
        with self._lock:
            modelo = self.modelo
        p, fatores = modelo.explicar(dados)
        lat = (time.perf_counter() - t0) * 1000

        corte = modelo.limiar if limiar is None else float(limiar)
        aprovado = p >= corte
        score = prob_para_score(p)
        rating, faixa = faixa_risco(score)

        i, n = TAXA_MENSAL_REF, dados["prazo_meses"]
        parcela = dados["valor_emprestimo"] * (i * (1 + i) ** n) / ((1 + i) ** n - 1)

        self.latencias.append(lat)
        self.total_predicoes += 1
        return {
            "nome_cliente": dados.get("nome_cliente", "Cliente Anônimo"),
            "decisao": "Aprovado" if aprovado else "Reprovado",
            "codigo_decisao": int(aprovado),
            "score_creditai": score,
            "rating": rating,
            "faixa_risco": faixa,
            "probabilidade_aprovacao": round(p, 4),
            "probabilidade_risco_inadimplencia": round(1 - p, 4),
            "limiar_politica": round(corte, 3),
            "parcela_estimada": round(parcela, 2),
            "comprometimento_renda": round(parcela / dados["renda_mensal"], 4),
            "fatores": fatores,
            "latencia_ms": round(lat, 3),
            "versao_modelo": modelo.meta.get("versao", "?"),
        }


model_service = CreditoService()
