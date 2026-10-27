"""Testes de contrato HTTP (200 / 400 / 422), explicabilidade e HUD."""

import time

import pytest


def test_health_modelo_em_ram(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status_servico"] == "online"
    assert body["modelo_carregado_em_ram"] is True
    assert body["metricas_referencia_teste"]["roc_auc"] > 0.75


def test_predict_payload_completo(client, cliente_valido):
    r = client.post("/predict", json=cliente_valido)
    assert r.status_code == 200
    b = r.json()
    assert b["decisao"] in {"Aprovado", "Reprovado"}
    assert 0 <= b["score_creditai"] <= 1000
    assert b["rating"] in {"AA", "A", "B", "C", "D"}
    assert abs(b["probabilidade_aprovacao"] + b["probabilidade_risco_inadimplencia"] - 1) < 1e-3
    assert len(b["fatores"]) == 11


@pytest.mark.parametrize("rota", ["/predict", "/v1/predict"])
def test_payload_minimo_retrocompativel(client, rota):
    r = client.post(rota, json={"idade": 42, "pontuacao_credito": 740})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"idade": 15, "pontuacao_credito": 990},
        {"idade": 30, "pontuacao_credito": 700, "tipo_moradia": "CASTELO"},
        {"idade": 30},
        {"idade": 30, "pontuacao_credito": 700, "campo_inexistente": 1},
        {"idade": 30, "pontuacao_credito": 700, "utilizacao_cartao": 1.7},
    ],
)
def test_422_pydantic(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "payload,trecho",
    [
        ({"idade": 20, "pontuacao_credito": 700, "tempo_emprego_anos": 15.0}, "idade laboral"),
        ({"idade": 80, "pontuacao_credito": 700, "prazo_meses": 120}, "85 anos"),
        ({"idade": 40, "pontuacao_credito": 700, "renda_mensal": 1000, "valor_emprestimo": 200000}, "120x"),
    ],
)
def test_400_regras_negocio(client, payload, trecho):
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    assert trecho in r.json()["detail"]


def test_limiar_politica_altera_decisao(client, cliente_valido):
    p = client.post("/predict", json=cliente_valido).json()["probabilidade_aprovacao"]
    abaixo = client.post(f"/predict?limiar={max(p - 0.05, 0.01):.2f}", json=cliente_valido).json()
    acima = client.post(f"/predict?limiar={min(p + 0.05, 0.99):.2f}", json=cliente_valido).json()
    assert abaixo["decisao"] == "Aprovado"
    assert acima["decisao"] == "Reprovado"


def test_monotonicidade_score_bureau(client, cliente_valido):
    """Restrição monotônica: subir o score de bureau nunca reduz a probabilidade."""
    probs = []
    for s in range(350, 851, 50):
        probs.append(client.post("/predict", json={**cliente_valido, "pontuacao_credito": s}).json()["probabilidade_aprovacao"])
    assert all(b >= a - 1e-9 for a, b in zip(probs, probs[1:]))
    assert probs[-1] > probs[0]


def test_inadimplencia_reduz_probabilidade(client, cliente_valido):
    ok = client.post("/predict", json=cliente_valido).json()["probabilidade_aprovacao"]
    ruim = client.post("/predict", json={**cliente_valido, "inadimplencia_previa": 1}).json()["probabilidade_aprovacao"]
    assert ruim <= ok


def test_bench_e_relatorio(client):
    r = client.get("/api/bench?n=50")
    assert r.status_code == 200
    b = r.json()
    assert sum(b["histograma_score"]["contagens"]) + b["bloqueados_regra_negocio"] == 50
    assert b["latencia_ms"]["p95"] < 500
    rep = client.get("/api/report").json()
    assert {"treino", "validacao", "teste"} <= set(rep["metricas"])


def test_paginas_hud(client):
    assert "CREDITAI ENGINE" in client.get("/").text
    assert "C.R.E.D.I.T" in client.get("/web/").text


def test_websocket_stream(client):
    with client.websocket_connect("/api/ws") as ws:
        msg = ws.receive_json()
        assert {"logs", "estado", "servico"} <= set(msg)
        assert msg["servico"]["modelo_em_ram"] is True


def test_treino_ao_vivo_com_hot_swap(client):
    r = client.post(
        "/api/train/start",
        json={"n_amostras": 3000, "max_iter": 60, "passo_iter": 20, "pausa_visual_s": 0, "regerar_dataset": True},
    )
    assert r.status_code == 200
    assert client.post("/api/train/start", json={}).status_code in (200, 409)
    for _ in range(300):
        st = client.get("/api/train/state").json()
        if st["status"] in {"COMPLETED", "STOPPED", "ERROR"}:
            break
        time.sleep(0.1)
    assert st["status"] == "COMPLETED", st
    assert st["historico"] and st["checkpoints"]
    assert client.get("/health").json()["modelo_carregado_em_ram"] is True
