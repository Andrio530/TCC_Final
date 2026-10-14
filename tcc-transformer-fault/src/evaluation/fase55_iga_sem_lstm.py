"""
Fase 5.5 — Ablação do LSTM (Track A), parte 1: IGA restrito às features SEM LSTM.

Executa o MESMO IGA (mesma população, gerações, seed, fitness CV 5-fold com
sample_weight) usado em train_hybrid.py, mas com o espaço de busca limitado às
9 features sem nenhuma coluna `lstm_*`. Isso permite comparar, com orçamento
de otimização idêntico:

    C1  : IGA com 17 features (obs + razões + LSTM)   [já existente]
    C5b : IGA com  9 features (obs + razões)          [este script]

Nada do conjunto de TESTE é usado aqui (o IGA só vê o treino via CV).

Saída: reports/metrics/fase55_track_a_iga_sem_lstm.json

Uso:
  python -m src.evaluation.fase55_iga_sem_lstm
"""

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import METRICS_DIR, RANDOM_SEED  # noqa: E402
from src.iga_xgboost.cromossomo import (  # noqa: E402
    FEATURES_TRACK_A_SEM_LSTM,
    decodificar,
)
from src.iga_xgboost.iga import IGA  # noqa: E402
from src.iga_xgboost.train_hybrid import (  # noqa: E402
    LABEL_MAP_A,
    _carregar_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TRACK = "A_SEM_LSTM"


def main() -> dict:
    df_train = _carregar_csv("katser_features_train.csv")
    X_train = df_train[FEATURES_TRACK_A_SEM_LSTM].values.astype(np.float32)
    y_train = np.array([LABEL_MAP_A[v] for v in df_train["fault_type"]], dtype=np.int32)

    counts = Counter(y_train)
    total, n_classes = len(y_train), len(counts)
    sample_weights = np.array([total / (n_classes * counts[yi]) for yi in y_train])

    logger.info("IGA sem LSTM — %d features: %s",
                len(FEATURES_TRACK_A_SEM_LSTM), FEATURES_TRACK_A_SEM_LSTM)
    t0 = time.time()
    iga = IGA(track=TRACK, X_train=X_train, y_train=y_train, seed=RANDOM_SEED,
              usar_smote=False, sample_weight=sample_weights)
    melhor = iga.evoluir()
    feats, hparams = decodificar(melhor, TRACK)

    resultado = {
        "descricao": "IGA restrito às features sem LSTM (ablação Fase 5.5)",
        "features_selecionadas": feats,
        "n_features": len(feats),
        "hiperparametros": hparams,
        "fitness_cv": float(iga.melhor_fitness),
        "n_geracoes": len(iga.historico_melhor),
        "historico_melhor": iga.historico_melhor,
        "historico_media": iga.historico_media,
        "seed": RANDOM_SEED,
        "tempo_segundos": time.time() - t0,
    }
    out = METRICS_DIR / "fase55_track_a_iga_sem_lstm.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    logger.info("Salvo em %s (fitness CV=%.4f, %d gerações, %.0fs)",
                out, iga.melhor_fitness, len(iga.historico_melhor),
                resultado["tempo_segundos"])
    return resultado


if __name__ == "__main__":
    main()
