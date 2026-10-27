"""
Motor de treino ao vivo (thread em background) com telemetria para o Visual Control HUD.

- Treino incremental do GBDT (warm_start) em passos → cada passo gera log + métricas.
- Checkpoint automático do "pico" de ROC-AUC na validação + early stopping por paciência.
- Ao final: restaura o melhor estado, calibra (isotônica) na validação, avalia no teste,
  serializa o artefato e faz HOT-SWAP do modelo servido pela API (sem reiniciar o servidor).
"""

from __future__ import annotations

import copy
import json
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import (
    CHECKPOINT_DIR,
    DATASET_PATH,
    METRICS_PATH,
    MODEL_CARD_PATH,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_VERSION,
    TrainConfig,
)
from app.ml.features import CATEGORICAL, NUMERIC_RAW, RAW_FEATURES, TARGET, gerar_dataset
from app.ml.model import CreditAIModel, PlattCalibrator, calcular_metricas, construir_pipeline, curva_calibracao


class TrainingEngine:
    LOG_MAX = 2000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.logs: deque[dict] = deque(maxlen=self.LOG_MAX)
        self._seq = 0
        self.config = TrainConfig()
        self.on_model_ready: Callable[[CreditAIModel], None] | None = None
        self._reset_state()

    # ------------------------------------------------------------------ estado
    def _reset_state(self) -> None:
        self.status = "IDLE"  # IDLE | RUNNING | STOPPING | COMPLETED | STOPPED | ERROR
        self.progresso = 0.0
        self.historico: list[dict] = []
        self.checkpoints: list[dict] = []
        self.pico: dict[str, Any] = {}
        self.metricas_finais: dict[str, Any] = {}
        self.inicio_ts: float | None = None
        self.fim_ts: float | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "progresso": round(self.progresso, 4),
                "config": self.config.to_dict(),
                "historico": list(self.historico),
                "checkpoints": list(self.checkpoints),
                "pico": dict(self.pico),
                "metricas_finais": dict(self.metricas_finais),
                "duracao_s": round((self.fim_ts or time.time()) - self.inicio_ts, 2) if self.inicio_ts else 0.0,
                "ultimo_seq": self._seq,
            }

    def logs_desde(self, seq: int) -> list[dict]:
        with self._lock:
            return [l for l in self.logs if l["seq"] > seq]

    def log(self, msg: str, canal: str = "TREINO") -> None:
        with self._lock:
            self._seq += 1
            self.logs.append(
                {"seq": self._seq, "ts": datetime.now().strftime("%H:%M:%S"), "canal": canal, "msg": msg}
            )
        print(f"[{canal}] {msg}", flush=True)

    def limpar_logs(self) -> None:
        with self._lock:
            self.logs.clear()

    def restaurar_do_relatorio(self) -> None:
        """Ao subir a API, repovoa o HUD com a telemetria do último treino salvo em disco."""
        if self.historico or not METRICS_PATH.exists():
            return
        try:
            rep = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            with self._lock:
                self.historico = rep.get("historico", [])
                self.metricas_finais = {"teste": rep["metricas"]["teste"]}
            self.log(f"Telemetria do último treino restaurada ({len(self.historico)} passos).", "SISTEMA")
        except Exception as exc:  # pragma: no cover
            self.log(f"Não foi possível restaurar telemetria: {exc}", "ERRO")

    @property
    def rodando(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ controle
    def iniciar(self, overrides: dict | None = None, bloqueante: bool = False) -> bool:
        if self.rodando:
            return False
        cfg = TrainConfig()
        for k, v in (overrides or {}).items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, type(getattr(cfg, k))(v))
        self.config = cfg
        self._stop.clear()
        with self._lock:
            self._reset_state()
            self.status = "RUNNING"
            self.inicio_ts = time.time()
        if bloqueante:
            self._executar()
        else:
            self._thread = threading.Thread(target=self._executar, name="creditai-trainer", daemon=True)
            self._thread.start()
        return True

    def parar(self) -> bool:
        if not self.rodando:
            return False
        self.status = "STOPPING"
        self._stop.set()
        self.log("Sinal de interrupção recebido — finalizando com o melhor checkpoint...", "SISTEMA")
        return True

    # ------------------------------------------------------------------ núcleo
    def _executar(self) -> None:
        cfg = self.config
        try:
            self.log(f"=== [HUD Worker] Iniciando treino {MODEL_NAME} v{MODEL_VERSION} ===", "SISTEMA")
            df = self._carregar_dataset(cfg)
            X, y = df[RAW_FEATURES], df[TARGET]
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=cfg.seed)
            X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=cfg.seed)
            self.log(
                f"Split estratificado 70/15/15 → treino={len(X_tr)} | validação={len(X_va)} | teste={len(X_te)} "
                f"| taxa base de bons pagadores={y.mean():.1%}",
                "SISTEMA",
            )

            pipe = construir_pipeline(cfg)
            melhor_auc, sem_melhora, melhor_pipe, melhor_iter = -1.0, 0, None, 0
            passos = list(range(cfg.passo_iter, cfg.max_iter + 1, cfg.passo_iter))
            amostra_lat = X_va.iloc[[0]]

            for k, n_iter in enumerate(passos, start=1):
                if self._stop.is_set():
                    self.log(f"Interrompido pelo operador no passo {n_iter}/{cfg.max_iter}.", "SISTEMA")
                    break
                t0 = time.perf_counter()
                pipe.set_params(clf__max_iter=n_iter)
                pipe.fit(X_tr, y_tr)
                dt = max(time.perf_counter() - t0, 1e-6)

                p_tr = pipe.predict_proba(X_tr)[:, 1]
                p_va = pipe.predict_proba(X_va)[:, 1]
                m_tr = calcular_metricas(y_tr, p_tr, cfg.limiar_decisao, "treino")
                m_va = calcular_metricas(y_va, p_va, cfg.limiar_decisao, "validacao")

                t1 = time.perf_counter()
                for _ in range(5):
                    pipe.predict_proba(amostra_lat)
                lat_ms = (time.perf_counter() - t1) / 5 * 1000

                throughput = len(X_tr) * cfg.passo_iter / dt
                registro = {
                    "passo": n_iter,
                    "max_iter": cfg.max_iter,
                    "throughput": round(throughput, 1),
                    "auc_treino": m_tr["roc_auc"],
                    "auc_val": m_va["roc_auc"],
                    "ks": m_va["ks"],
                    "gini": m_va["gini"],
                    "loss_treino": m_tr["log_loss"],
                    "loss_val": m_va["log_loss"],
                    "brier": m_va["brier"],
                    "ece": m_va["ece"],
                    "acuracia": m_va["acuracia"],
                    "f1": m_va["f1_score"],
                    "gap": round(m_tr["roc_auc"] - m_va["roc_auc"], 4),
                    "latencia_ms": round(lat_ms, 2),
                }
                with self._lock:
                    self.historico.append(registro)
                    self.progresso = k / len(passos)

                self.log(
                    f"[Iter {n_iter}/{cfg.max_iter}] AUC val: {m_va['roc_auc']:.4f} | KS: {m_va['ks']:.3f} | "
                    f"Gini: {m_va['gini']:.3f} | LogLoss: {m_va['log_loss']:.4f} | ECE: {m_va['ece']:.3f} | "
                    f"Gap treino/val: {registro['gap']:+.3f} | Throughput: {throughput/1000:.1f}k amostras/s | "
                    f"Latência: {lat_ms:.2f}ms",
                    "TREINO",
                )

                if m_va["roc_auc"] > melhor_auc + 1e-4:
                    melhor_auc, melhor_iter, sem_melhora = m_va["roc_auc"], n_iter, 0
                    melhor_pipe = copy.deepcopy(pipe)
                    nome = f"ckpt_iter{n_iter:04d}.joblib"
                    joblib.dump(melhor_pipe, CHECKPOINT_DIR / nome, compress=3)
                    ck = {"arquivo": nome, "passo": n_iter, "auc_val": m_va["roc_auc"], "ks": m_va["ks"],
                          "ts": datetime.now().strftime("%H:%M:%S")}
                    with self._lock:
                        self.checkpoints.append(ck)
                        self.pico = ck
                    self.log(f"Novo pico de AUC {m_va['roc_auc']:.4f} → checkpoint persistido em: {nome}", "CHECKPOINT")
                else:
                    sem_melhora += 1
                    if sem_melhora >= cfg.paciencia:
                        self.log(
                            f"[EARLY STOPPING] {cfg.paciencia} passos sem melhora. Pico em iter {melhor_iter} "
                            f"(AUC {melhor_auc:.4f}).",
                            "VALID",
                        )
                        break
                if cfg.pausa_visual_s > 0:
                    time.sleep(cfg.pausa_visual_s)

            if melhor_pipe is None:
                raise RuntimeError("Nenhum passo de treino concluído — nada para salvar.")
            self._finalizar(cfg, melhor_pipe, melhor_iter, X_tr, X_va, y_va, X_te, y_te, y_tr)
            with self._lock:
                self.status = "STOPPED" if self._stop.is_set() else "COMPLETED"
                self.progresso = 1.0
        except Exception as exc:  # pragma: no cover - caminho de erro exibido no HUD
            self.log(f"Falha no treino: {exc}", "ERRO")
            self.log(traceback.format_exc().splitlines()[-1], "ERRO")
            with self._lock:
                self.status = "ERROR"
        finally:
            self.fim_ts = time.time()
            self.log(f"Tarefa finalizada com status {self.status}.", "SISTEMA")

    def _carregar_dataset(self, cfg: TrainConfig) -> pd.DataFrame:
        if DATASET_PATH.exists() and not cfg.regerar_dataset:
            df = pd.read_csv(DATASET_PATH)
            if set(RAW_FEATURES + [TARGET]).issubset(df.columns) and len(df) == cfg.n_amostras:
                self.log(f"Dataset carregado de data/{DATASET_PATH.name} ({len(df)} registros).", "SISTEMA")
                return df
        df = gerar_dataset(cfg.n_amostras, cfg.seed)
        df.to_csv(DATASET_PATH, index=False)
        self.log(f"Dataset sintético gerado e salvo em data/{DATASET_PATH.name} ({len(df)} registros).", "SISTEMA")
        return df

    def _finalizar(self, cfg, pipe, melhor_iter, X_tr, X_va, y_va, X_te, y_te, y_tr) -> None:
        self.log(f"[Best State Restored] Restaurando pipeline do pico (iter {melhor_iter}).", "CHECKPOINT")
        p_va_raw = pipe.predict_proba(X_va)[:, 1]
        iso = PlattCalibrator().fit(p_va_raw, y_va)
        self.log("Calibração Platt (sigmoide) ajustada na validação (probabilidades = taxas reais de adimplência).", "VALID")

        perfil = {}
        for c in NUMERIC_RAW:
            perfil[c] = float(np.nanmedian(X_tr[c]))
        for c in CATEGORICAL:
            perfil[c] = str(X_tr[c].mode().iloc[0])

        modelo = CreditAIModel(pipeline=pipe, calibrador=iso, perfil_referencia=perfil, limiar=cfg.limiar_decisao)
        p_te = modelo.predict_proba(X_te)
        p_te_raw = pipe.predict_proba(X_te)[:, 1]
        m_te = calcular_metricas(y_te, p_te, cfg.limiar_decisao, "teste")
        m_te_raw = calcular_metricas(y_te, p_te_raw, cfg.limiar_decisao, "teste_sem_calibracao")
        m_va = calcular_metricas(y_va, modelo.predict_proba(X_va), cfg.limiar_decisao, "validacao")
        m_tr = calcular_metricas(y_tr, modelo.predict_proba(X_tr), cfg.limiar_decisao, "treino")

        modelo.meta = {
            "nome": MODEL_NAME,
            "versao": MODEL_VERSION,
            "algoritmo": "HistGradientBoosting (monotônico) + Calibração Platt",
            "melhor_iteracao": melhor_iter,
            "treinado_em": datetime.now().isoformat(timespec="seconds"),
            "metricas_teste": m_te,
        }
        joblib.dump(modelo, MODEL_PATH, compress=3)
        tamanho_mb = MODEL_PATH.stat().st_size / 1e6

        relatorio = {
            "modelo": modelo.meta,
            "config": cfg.to_dict(),
            "metricas": {"treino": m_tr, "validacao": m_va, "teste": m_te, "teste_sem_calibracao": m_te_raw},
            "curva_calibracao_teste": curva_calibracao(y_te, p_te),
            "historico": self.historico,
        }
        METRICS_PATH.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
        MODEL_CARD_PATH.write_text(
            json.dumps(
                {
                    **modelo.meta,
                    "features": RAW_FEATURES,
                    "uso_pretendido": "Apoio à decisão de concessão de crédito pessoal (dados sintéticos/educacional).",
                    "limitacoes": "Base sintética; recalibrar e auditar viés antes de qualquer uso real.",
                    "tamanho_mb": round(tamanho_mb, 2),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        with self._lock:
            self.metricas_finais = {"teste": m_te, "teste_sem_calibracao": m_te_raw, "tamanho_mb": round(tamanho_mb, 2)}

        self.log(
            f"[TESTE] AUC: {m_te['roc_auc']:.4f} | KS: {m_te['ks']:.3f} | Gini: {m_te['gini']:.3f} | "
            f"Brier: {m_te['brier']:.4f} | ECE: {m_te_raw['ece']:.3f} → {m_te['ece']:.3f} (calibrado) | "
            f"F1: {m_te['f1_score']:.3f}",
            "VALID",
        )
        self.log(f"Artefato salvo: models/{MODEL_PATH.name} ({tamanho_mb:.2f} MB)", "CHECKPOINT")
        if self.on_model_ready:
            self.on_model_ready(modelo)
            self.log("[Hot-Swap] Novo modelo carregado em RAM — API servindo a nova versão sem reinício.", "SISTEMA")
        self.log(f"=== [HUD Worker] Treinamento concluído! AUC teste final: {m_te['roc_auc']:.4f} ===", "SISTEMA")


engine = TrainingEngine()
