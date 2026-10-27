"""
Treino offline via linha de comando (mesmo motor usado pelo HUD).

    python train_pipeline.py                     # configuração padrão
    python train_pipeline.py --max-iter 600 --lr 0.04 --regerar
"""

import argparse

from app.services.training_engine import engine


def executar_treinamento(**overrides) -> dict:
    overrides.setdefault("pausa_visual_s", 0.0)
    engine.iniciar(overrides, bloqueante=True)
    return engine.snapshot()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treino do CreditAI (GBDT monotônico + calibração)")
    ap.add_argument("--amostras", type=int, dest="n_amostras")
    ap.add_argument("--max-iter", type=int, dest="max_iter")
    ap.add_argument("--passo", type=int, dest="passo_iter")
    ap.add_argument("--lr", type=float, dest="learning_rate")
    ap.add_argument("--regerar", action="store_true", dest="regerar_dataset")
    args = {k: v for k, v in vars(ap.parse_args()).items() if v not in (None, False)}
    snap = executar_treinamento(**args)
    print(f"\nStatus final: {snap['status']} | duração {snap['duracao_s']}s")
