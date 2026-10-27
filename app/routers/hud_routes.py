"""Rotas do Visual Control HUD: controle de treino, logs, relatórios, benchmark e WebSocket."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from app.core.config import METRICS_PATH
from app.ml.features import gerar_dataset
from app.ml.model import prob_para_score
from app.schemas.cliente import TreinoRequest
from app.services.credito_service import RegraNegocioError, model_service
from app.services.training_engine import engine

hud_router = APIRouter(prefix="/api", tags=["Visual Control HUD"])


@hud_router.post("/train/start")
def iniciar_treino(req: TreinoRequest | None = None):
    if not engine.iniciar((req or TreinoRequest()).model_dump(exclude_none=True)):
        raise HTTPException(409, "Já existe um treino em execução.")
    return {"ok": True, "status": engine.status, "config": engine.config.to_dict()}


@hud_router.post("/train/stop")
def parar_treino():
    return {"ok": engine.parar(), "status": engine.status}


@hud_router.get("/train/state")
def estado_treino():
    return engine.snapshot()


@hud_router.get("/logs")
def logs(desde: int = 0):
    return engine.logs_desde(desde)


@hud_router.delete("/logs")
def limpar_logs():
    engine.limpar_logs()
    return {"ok": True}


@hud_router.get("/report")
def relatorio():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "Relatório ainda não gerado — execute um treino.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@hud_router.post("/model/reload")
def recarregar_modelo():
    if not model_service.model_path.exists():
        raise HTTPException(404, "Artefato não encontrado em disco.")
    model_service.carregar_modelo_em_ram()
    engine.log("[Sync] Artefato recarregado do disco para a RAM.", "SISTEMA")
    return {"ok": True, "meta": model_service.modelo.meta}


@hud_router.get("/bench")
def benchmark(n: int = Query(200, ge=10, le=2000), limiar: float | None = Query(None, gt=0, lt=1)):
    """Stress test: n perfis sintéticos inéditos → distribuição de latência, score e aprovação."""
    if not model_service.carregado:
        raise HTTPException(503, "Modelo não carregado.")
    df = gerar_dataset(n, seed=int(time.time()) % 10_000).drop(columns=["aprovado"])
    df = df.fillna({"renda_mensal": 4000.0, "tempo_emprego_anos": 0.0, "utilizacao_cartao": 0.3})
    registros = df.to_dict(orient="records")
    validos = []
    for row in registros:
        try:
            model_service.validar_regras_negocio(row)
            validos.append(row)
        except RegraNegocioError:
            pass
    bloqueios = n - len(validos)
    if not validos:
        raise HTTPException(422, "Nenhum perfil válido gerado para o benchmark.")
    # Latência ponta-a-ponta (inferência + explicabilidade) em uma amostra de requisições unitárias
    lat = [model_service.prever({**r, "nome_cliente": "bench"}, limiar)["latencia_ms"] for r in validos[:60]]
    # Scoring vetorizado do lote completo (modo batch)
    t0 = time.perf_counter()
    probs = model_service.modelo.predict_proba(pd.DataFrame(validos))
    batch_ms = (time.perf_counter() - t0) * 1000
    corte = model_service.modelo.limiar if limiar is None else limiar
    scores = [prob_para_score(p) for p in probs]
    aprov = int((probs >= corte).sum())
    lat_a = np.array(lat)
    hist, edges = np.histogram(scores, bins=20, range=(0, 1000))
    validos_n = max(len(validos), 1)
    return {
        "n": n,
        "bloqueados_regra_negocio": bloqueios,
        "taxa_aprovacao": round(aprov / validos_n, 4),
        "lote_vetorizado_ms": round(batch_ms, 2),
        "latencia_ms": {
            "p50": round(float(np.percentile(lat_a, 50)), 3),
            "p95": round(float(np.percentile(lat_a, 95)), 3),
            "p99": round(float(np.percentile(lat_a, 99)), 3),
            "media": round(float(lat_a.mean()), 3),
        },
        "score_medio": round(float(np.mean(scores)), 1),
        "histograma_score": {"contagens": hist.tolist(), "bordas": edges.tolist()},
    }


@hud_router.websocket("/ws")
async def hud_stream(ws: WebSocket):
    """Stream contínuo (4 Hz) de logs novos + snapshot do motor de treino."""
    await ws.accept()
    cursor = int(ws.query_params.get("desde", 0))
    try:
        while True:
            novos = engine.logs_desde(cursor)
            if novos:
                cursor = novos[-1]["seq"]
            snap = engine.snapshot()
            await ws.send_json(
                {
                    "logs": novos,
                    "estado": snap,
                    "servico": {
                        "modelo_em_ram": model_service.carregado,
                        "latencia_p50_ms": model_service.latencia_p50(),
                        "total_predicoes": model_service.total_predicoes,
                    },
                }
            )
            await asyncio.sleep(0.25)
    except (WebSocketDisconnect, RuntimeError):
        return
