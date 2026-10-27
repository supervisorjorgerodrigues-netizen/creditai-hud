# ⚡ CreditAI Engine — Visual Control HUD (Score de Crédito)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikitlearn&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-23%20testes-0A9EDC?logo=pytest&logoColor=white)

Evolução do projeto `credito-ml-api` para uma plataforma de **score de crédito com treino ao vivo**,
inspirada no *System 1 Engine HUD*: um painel de controle neon onde você **inicia o treino, acompanha
métricas em tempo real via WebSocket, inspeciona checkpoints e avalia proponentes** com score 0–1000,
rating e explicação por atributo.

> Situação de Aprendizagem — Análise Preditiva de Dados & IA (SCTEC) · Autor: Jorge Luiz Susin Rodrigues

---

## 🚀 Execução rápida

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python run.py                        # abre http://127.0.0.1:8050
```

Na primeira execução o `lifespan` **gera o dataset e treina o modelo automaticamente** (~7 s).

| Página | URL | O que faz |
|---|---|---|
| 🏋 HUD de Treinamento | `http://127.0.0.1:8050/` | INICIAR/PARAR treino, 6 cards de métricas com sparklines, terminal de logs filtrável, curvas AUC/LogLoss/ECE, checkpoints, relatório de teste, matriz de confusão e diagrama de confiabilidade, mini-HUD flutuante |
| 🎮 Avaliação | `http://127.0.0.1:8050/web/` | Simulador com gauge 0–1000, decisão, fatores explicativos, política de limiar, laboratório HTTP 200/422/400, stress test e telemetria de latência |
| 📘 Swagger | `http://127.0.0.1:8050/docs` | Documentação interativa da API |

Atalhos na Avaliação: `Espaço` avaliar · `1-4` perfis · `B` stress test · `R` reset.

Outros comandos:

```bash
python train_pipeline.py --max-iter 600 --lr 0.03 --regerar   # treino offline (mesmo motor do HUD)
uvicorn main:app --port 8050                                   # servidor sem abrir o navegador
pytest -v                                                      # 23 testes (artefatos isolados em diretório temporário)
```

---

## 🧠 O que há de "última geração" no modelo

| Recurso | Por quê |
|---|---|
| **HistGradientBoostingClassifier** | GBDT nativo do scikit-learn (estilo LightGBM), rápido em CPU e com suporte nativo a valores ausentes |
| **Restrições monotônicas** | Score de bureau/renda maiores **nunca** pioram a decisão; inadimplência, consultas, uso do cartão e comprometimento **nunca** melhoram — exigência regulatória de coerência (testado em `test_monotonicidade_score_bureau`) |
| **Engenharia de atributos dentro do Pipeline** | Parcela (Tabela Price), comprometimento de renda, LTI e score por consulta — sem *data leakage* |
| **Treino incremental (warm start) + checkpoint do pico** | Telemetria a cada N árvores, early stopping por paciência e restauração do melhor estado |
| **Calibração Platt** | Probabilidades viram taxas reais de adimplência (ECE monitorado); score não satura em 0/1000 |
| **Métricas de risco de crédito** | ROC-AUC, **KS**, **Gini**, LogLoss, Brier, ECE, além de acurácia/precisão/recall/F1 |
| **Score 0–1000 por PDO** | Escala de *scorecard* (PDO=60, 600 pontos = odds 3:1) com ratings AA → D |
| **Explicabilidade por atributo** | Δ de probabilidade ao substituir cada atributo pelo perfil mediano (um único batch, poucos ms) |
| **Hot-swap** | Um treino disparado pelo HUD substitui o modelo servido **sem reiniciar** a API |

### Métricas típicas (12.000 registros sintéticos, split estratificado 70/15/15)

| Partição | ROC-AUC | KS | Gini | Brier | ECE | F1 |
|---|---|---|---|---|---|---|
| Validação | ~0.818 | ~0.48 | ~0.636 | ~0.145 | ~0.02 | ~0.865 |
| **Teste** | **~0.813** | **~0.48** | **~0.626** | **~0.151** | **~0.027** | **~0.849** |

> O ruído da base sintética é proposital (simula informação não observável); AUC ≈ 0.81 é realista
> para score de crédito pessoal. Valores exatos ficam em `models/relatorio_metricas.json`.

---

## 🏗 Arquitetura

```text
creditai-hud/
├── app/
│   ├── core/config.py              # caminhos relativos, TrainConfig (hiperparâmetros, env CREDITAI_*)
│   ├── ml/features.py              # dataset sintético causal + engenharia de atributos + monotonicidade
│   ├── ml/model.py                 # pipeline GBDT, métricas (KS/Gini/ECE), Platt, score PDO, CreditAIModel
│   ├── schemas/cliente.py          # contratos Pydantic v2 (extra="forbid", Enums, Field)
│   ├── services/credito_service.py # singleton de inferência em RAM, regras de negócio (400), latência
│   ├── services/training_engine.py # motor de treino em thread: logs, checkpoints, early stop, hot-swap
│   └── routers/
│       ├── predict_routes.py       # GET /health · POST /predict · POST /v1/predict
│       └── hud_routes.py           # /api/train/* · /api/logs · /api/report · /api/bench · WS /api/ws
├── static/                         # HUD: hud.html/js (treino), avaliacao.html/js, hud.css, Chart.js local
├── tests/                          # test_api.py (HTTP/WS/treino) · test_pipeline.py (núcleo ML)
├── main.py                         # FastAPI + lifespan (bootstrap de treino e carga em RAM)
├── train_pipeline.py               # CLI de treino
└── run.py                          # atalho: sobe o servidor e abre o navegador
```

Fluxo ao vivo: o botão **INICIAR** chama `POST /api/train/start` → o `TrainingEngine` treina em uma
thread → o navegador recebe logs + snapshot a 4 Hz pelo WebSocket `/api/ws` → ao terminar, o
modelo calibrado é salvo e injetado no `CreditoService` (hot-swap).

---

## 🔌 API

```bash
curl -X POST "http://127.0.0.1:8050/predict?limiar=0.5" -H "Content-Type: application/json" -d '{
  "nome_cliente": "Carlos Eduardo Santos", "idade": 35, "pontuacao_credito": 720,
  "renda_mensal": 7500, "valor_emprestimo": 18000, "prazo_meses": 24,
  "tempo_emprego_anos": 5, "inadimplencia_previa": 0, "consultas_6m": 1,
  "utilizacao_cartao": 0.3, "tipo_moradia": "PROPRIA", "finalidade_emprestimo": "NEGOCIOS"
}'
```

Resposta (resumida): `decisao`, `score_creditai` (0–1000), `rating`, `faixa_risco`,
`probabilidade_aprovacao`, `parcela_estimada`, `comprometimento_renda`, `fatores[]`, `latencia_ms`.

| Cenário | Payload | Status |
|---|---|---|
| Contrato mínimo retrocompatível | `{"idade": 42, "pontuacao_credito": 740}` | **200** |
| Limites Pydantic | `{"idade": 15, "pontuacao_credito": 990}` | **422** |
| Campo desconhecido / enum inválido | `{"idade": 30, "pontuacao_credito": 700, "tipo_moradia": "CASTELO"}` | **422** |
| Emprego > idade laboral | `{"idade": 20, "pontuacao_credito": 700, "tempo_emprego_anos": 15}` | **400** |
| Idade ao fim do contrato > 85 | `{"idade": 80, "pontuacao_credito": 700, "prazo_meses": 120}` | **400** |
| Valor > 120× renda | `{"idade": 40, "pontuacao_credito": 700, "renda_mensal": 1000, "valor_emprestimo": 200000}` | **400** |

Endpoints do HUD: `POST /api/train/start` (corpo opcional com hiperparâmetros), `POST /api/train/stop`,
`GET /api/train/state`, `GET|DELETE /api/logs`, `GET /api/report`, `POST /api/model/reload`,
`GET /api/bench?n=500`, `WS /api/ws`.

---

## ⚠️ Limitações

Base **sintética** para fins educacionais. Antes de qualquer uso real: dados históricos de safra,
validação fora do tempo (out-of-time), auditoria de viés/fairness e conformidade com LGPD e
normas do Banco Central.
