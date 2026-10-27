"""
====================================================================================
CreditAI HUD — Ponto de entrada FastAPI + Lifespan
====================================================================================
- Lifespan: treina automaticamente se o artefato não existir e carrega o modelo na RAM
  UMA única vez; libera no shutdown.
- Motor de treino com hot-swap: um novo treino disparado pelo HUD substitui o modelo
  servido sem reiniciar o processo.
- Serve o Visual Control HUD:  /      → Treinamento (logs & status ao vivo)
                               /web/  → Avaliação (simulador & scores)
Execução:  python run.py   ou   uvicorn main:app --port 8050
====================================================================================
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import MODEL_VERSION, STATIC_DIR
from app.routers.hud_routes import hud_router
from app.routers.predict_routes import health_router, predict_router
from app.services.credito_service import model_service
from app.services.training_engine import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.on_model_ready = model_service.substituir_modelo
    if not model_service.model_path.exists():
        engine.log("Artefato não encontrado em disco — executando treino inicial (bootstrap)...", "SISTEMA")
        engine.iniciar({"pausa_visual_s": 0.0}, bloqueante=True)
    model_service.carregar_modelo_em_ram()
    engine.restaurar_do_relatorio()
    app.state.model_service = model_service
    engine.log(f"[LIFESPAN] Modelo {model_service.modelo.meta.get('nome')} carregado em RAM. API pronta.", "SISTEMA")
    yield
    engine.parar()
    model_service.descarregar_modelo_da_ram()


app = FastAPI(
    title="CreditAI HUD — API de Score de Crédito",
    description=(
        "Microsserviço de score de crédito com GBDT monotônico calibrado, explicabilidade por "
        "atributo, motor de treino ao vivo (WebSocket) e Visual Control HUD."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(hud_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def hud_treino():
    return FileResponse(STATIC_DIR / "hud.html")


@app.get("/web", include_in_schema=False)
def _web_redirect():
    return RedirectResponse("/web/")


@app.get("/web/", include_in_schema=False)
def hud_avaliacao():
    return FileResponse(STATIC_DIR / "avaliacao.html")
