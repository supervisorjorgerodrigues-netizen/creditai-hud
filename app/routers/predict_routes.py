"""Rotas REST de saúde e inferência (+ rota versionada /v1)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.cliente import DadosClienteInput, HealthResponse, PredicaoCreditoOutput
from app.services.credito_service import ModeloIndisponivelError, RegraNegocioError, model_service
from app.services.training_engine import engine

health_router = APIRouter(tags=["Monitoramento"])
predict_router = APIRouter(tags=["Inferência"])


@health_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    meta = model_service.modelo.meta if model_service.carregado else {}
    return HealthResponse(
        status_servico="online",
        modelo_carregado_em_ram=model_service.carregado,
        modelo=meta.get("nome"),
        algoritmo=meta.get("algoritmo"),
        versao_modelo=meta.get("versao"),
        treinado_em=meta.get("treinado_em"),
        status_treino=engine.status,
        metricas_referencia_teste={
            k: v for k, v in meta.get("metricas_teste", {}).items() if k != "matriz_confusao"
        },
        latencia_p50_ms=model_service.latencia_p50(),
        total_predicoes=model_service.total_predicoes,
    )


def _prever(payload: DadosClienteInput, limiar: Optional[float]) -> dict:
    try:
        return model_service.prever(payload.model_dump(mode="json"), limiar)
    except RegraNegocioError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ModeloIndisponivelError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@predict_router.post("/predict", response_model=PredicaoCreditoOutput)
def predict(
    payload: DadosClienteInput,
    limiar: Optional[float] = Query(None, gt=0, lt=1, description="Override do ponto de corte da política"),
):
    return _prever(payload, limiar)


@predict_router.post("/v1/predict", response_model=PredicaoCreditoOutput)
def predict_v1(payload: DadosClienteInput, limiar: Optional[float] = Query(None, gt=0, lt=1)):
    return _prever(payload, limiar)
