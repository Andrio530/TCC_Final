"""
Fase 5 — Track A: treinamento do IGA-XGBoost na base Katser.

Fluxo completo do sistema híbrido:
  1. Carrega os CSVs de features gerados na Fase 4A
     (katser_features_{train,val,test}.csv)
  2. Codifica os labels (str → int) com mapeamento fixo
  3. Aplica SMOTE no treino para tratar o desbalanceamento de classes
     (Normal ≈ 81% das séries Katser)
  4. Executa o IGA para encontrar (features ótimas, hiperparâmetros ótimos)
  5. Treina o XGBoost final com os melhores parâmetros e avalia no teste
  6. Executa o XGBoost com parâmetros default para comparação (C2 do plano)
  7. Repete o treino final com 5 seeds para obter média ± desvio-padrão
  8. Salva métricas, modelo e histórico de evolução

Saídas (reports/metrics/ e data/processed/):
  iga_track_a_results.json    — métricas completas (IGA + default)
  iga_evolution_track_a.json  — histórico de fitness por geração
  iga_best_model_track_a.pkl  — modelo XGBoost final (melhor seed)
"""

import json
import logging
import pickle
import sys
from pathlib import Path

from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DATA_PROCESSED,
    METRICS_DIR,
    RANDOM_SEED,
    SEEDS_AVALIACAO,
)
from src.iga_xgboost.cromossomo import FEATURES_TRACK_A, decodificar  # noqa: E402
from src.iga_xgboost.iga import IGA  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TRACK = "A"

# Mapeamento fixo de labels — garante consistência entre splits
LABEL_MAP_A: dict[str, int] = {"Normal": 0, "PD": 1, "D1": 2, "T1": 3}
LABEL_MAP_A_INV: dict[int, str] = {v: k for k, v in LABEL_MAP_A.items()}

# Hiperparâmetros default para comparação (Experimento C2)
HPARAMS_DEFAULT = {
    "n_estimators":     100,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.9,
    "colsample_bytree": 1.0,
    "gamma":            0.0,
    "min_child_weight": 1,
    "reg_alpha":        0.0,
    "reg_lambda":       1.0,
}


# ---------------------------------------------------------------------------
# 1. Carga de dados
# ---------------------------------------------------------------------------

def _carregar_csv(nome: str) -> pd.DataFrame:
    caminho = DATA_PROCESSED / nome
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo '{caminho}' não encontrado. "
            "Execute src/feature_engineering/run_feature_engineering.py antes."
        )
    return pd.read_csv(caminho)


def carregar_dados() -> tuple:
    """Carrega e separa features/labels dos três splits.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Arrays prontos para uso.
    """
    df_train = _carregar_csv("katser_features_train.csv")
    df_val   = _carregar_csv("katser_features_val.csv")
    df_test  = _carregar_csv("katser_features_test.csv")

    def _separar(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[FEATURES_TRACK_A].values.astype(np.float32)
        y = np.array([LABEL_MAP_A[v] for v in df["fault_type"]], dtype=np.int32)
        return X, y

    X_train, y_train = _separar(df_train)
    X_val,   y_val   = _separar(df_val)
    X_test,  y_test  = _separar(df_test)

    logger.info(
        "Track A — treino: %s | val: %s | teste: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )
    logger.info(
        "Distribuição treino (pré-SMOTE): %s",
        {LABEL_MAP_A_INV[k]: int(v) for k, v in
         zip(*np.unique(y_train, return_counts=True))},
    )
    return X_train, y_train, X_val, y_val, X_test, y_test





# ---------------------------------------------------------------------------
# 3. Treino e avaliação do XGBoost
# ---------------------------------------------------------------------------

def _treinar_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hparams: dict,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> XGBClassifier:
    """Treina um XGBClassifier com os parâmetros fornecidos."""
    modelo = XGBClassifier(
        **hparams,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    modelo.fit(X_train, y_train, sample_weight=sample_weight)
    return modelo


def _avaliar_modelo(
    modelo: XGBClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    classes: list[str],
) -> dict:
    """Calcula métricas de classificação para um split."""
    y_pred = modelo.predict(X)
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro",
                                          zero_division=0)),
        "report":          classification_report(
                               y_true, y_pred,
                               target_names=classes,
                               zero_division=0,
                               output_dict=True,
                           ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classes":         classes,
    }


# ---------------------------------------------------------------------------
# 4. Avaliação multi-seed
# ---------------------------------------------------------------------------

def _avaliar_multi_seed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hparams: dict,
    features_idx: list[int],
    classes: list[str],
    sample_weight: np.ndarray | None = None,
) -> dict:
    """Treina 5 vezes com seeds diferentes e retorna média ± desvio."""
    f1_scores: list[float] = []
    acc_scores: list[float] = []

    for seed in SEEDS_AVALIACAO:
        m = _treinar_xgboost(X_train[:, features_idx], y_train, hparams, seed, sample_weight=sample_weight)
        y_pred = m.predict(X_test[:, features_idx])
        f1_scores.append(float(f1_score(y_test, y_pred, average="macro",
                                        zero_division=0)))
        acc_scores.append(float(accuracy_score(y_test, y_pred)))

    return {
        "f1_macro_mean":  float(np.mean(f1_scores)),
        "f1_macro_std":   float(np.std(f1_scores)),
        "accuracy_mean":  float(np.mean(acc_scores)),
        "accuracy_std":   float(np.std(acc_scores)),
        "f1_por_seed":    f1_scores,
        "seeds":          SEEDS_AVALIACAO,
    }


# ---------------------------------------------------------------------------
# 5. Pipeline principal
# ---------------------------------------------------------------------------

def main() -> dict:
    """Executa o pipeline completo do Track A e retorna as métricas."""
    logger.info("=" * 60)
    logger.info("Fase 5 — Track A (Katser, 4 classes, sistema híbrido)")
    logger.info("=" * 60)

    classes = list(LABEL_MAP_A.keys())  # ['Normal', 'PD', 'D1', 'T1']

    # 1. Carrega dados
    X_train, y_train, X_val, y_val, X_test, y_test = carregar_dados()

    # 2. Pesos das classes (sample_weight) em vez de SMOTE
    counts = Counter(y_train)
    total = len(y_train)
    n_classes = len(counts)
    sample_weights = np.array([total / (n_classes * counts[yi]) for yi in y_train])

    # -------------------------------------------------------------------
    # Experimento C1 — IGA-XGBoost (otimizado)
    # -------------------------------------------------------------------
    logger.info("Executando IGA (Track A)...")
    iga = IGA(
        track=TRACK, X_train=X_train, y_train=y_train, seed=RANDOM_SEED,
        usar_smote=False, sample_weight=sample_weights
    )
    melhor_crom = iga.evoluir()

    features_ativas, hparams_iga = decodificar(melhor_crom, TRACK)
    features_idx_iga = [FEATURES_TRACK_A.index(f) for f in features_ativas]

    logger.info("Treinando modelo final (C1 — IGA)...")
    
    modelo_iga = _treinar_xgboost(
        X_train[:, features_idx_iga], y_train, hparams_iga, RANDOM_SEED,
        sample_weight=sample_weights
    )

    metricas_c1_val  = _avaliar_modelo(modelo_iga, X_val[:, features_idx_iga],
                                       y_val, classes)
    metricas_c1_test = _avaliar_modelo(modelo_iga, X_test[:, features_idx_iga],
                                       y_test, classes)
    multiseed_c1 = _avaliar_multi_seed(X_train, y_train, X_test, y_test,
                                       hparams_iga, features_idx_iga, classes,
                                       sample_weight=sample_weights)

    logger.info(
        "C1 (IGA) — Val F1=%.4f | Teste F1=%.4f ± %.4f",
        metricas_c1_val["f1_macro"],
        multiseed_c1["f1_macro_mean"],
        multiseed_c1["f1_macro_std"],
    )

    # -------------------------------------------------------------------
    # Experimento C2 — XGBoost default (sem IGA)
    # -------------------------------------------------------------------
    logger.info("Treinando modelo default (C2 — sem IGA)...")
    todas_features_idx = list(range(len(FEATURES_TRACK_A)))
    modelo_default = _treinar_xgboost(X_train, y_train, HPARAMS_DEFAULT, RANDOM_SEED, sample_weight=sample_weights)
    metricas_c2_val  = _avaliar_modelo(modelo_default, X_val, y_val, classes)
    metricas_c2_test = _avaliar_modelo(modelo_default, X_test, y_test, classes)
    multiseed_c2 = _avaliar_multi_seed(X_train, y_train, X_test, y_test,
                                       HPARAMS_DEFAULT, todas_features_idx, classes, sample_weight=sample_weights)

    logger.info(
        "C2 (default) — Val F1=%.4f | Teste F1=%.4f ± %.4f",
        metricas_c2_val["f1_macro"],
        multiseed_c2["f1_macro_mean"],
        multiseed_c2["f1_macro_std"],
    )

    # -------------------------------------------------------------------
    # Salvar resultados
    # -------------------------------------------------------------------
    resultados = {
        "track": "A",
        "descricao": "Sistema híbrido completo (LSTM → XGBoost) na base Katser",
        "classes": classes,
        "C1_IGA": {
            "features_selecionadas": features_ativas,
            "n_features": len(features_ativas),
            "hiperparametros": hparams_iga,
            "fitness_cv": float(iga.melhor_fitness),
            "val":       metricas_c1_val,
            "test":      metricas_c1_test,
            "multiseed": multiseed_c1,
        },
        "C2_default": {
            "features_selecionadas": FEATURES_TRACK_A,
            "n_features": len(FEATURES_TRACK_A),
            "hiperparametros": HPARAMS_DEFAULT,
            "val":       metricas_c2_val,
            "test":      metricas_c2_test,
            "multiseed": multiseed_c2,
        },
    }

    # JSON de métricas
    out_metrics = METRICS_DIR / "iga_track_a_results.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    logger.info("Métricas salvas em '%s'.", out_metrics)

    # JSON de histórico de evolução
    out_evolucao = METRICS_DIR / "iga_evolution_track_a.json"
    with open(out_evolucao, "w", encoding="utf-8") as f:
        json.dump({
            "historico_melhor": iga.historico_melhor,
            "historico_media":  iga.historico_media,
        }, f, indent=2)
    logger.info("Histórico de evolução salvo em '%s'.", out_evolucao)

    # Modelo final
    out_modelo = DATA_PROCESSED / "iga_best_model_track_a.pkl"
    with open(out_modelo, "wb") as f:
        pickle.dump({
            "modelo": modelo_iga,
            "features_ativas": features_ativas,
            "features_idx": features_idx_iga,
            "label_map": LABEL_MAP_A,
        }, f)
    logger.info("Modelo salvo em '%s'.", out_modelo)

    logger.info("=" * 60)
    logger.info("Track A concluído.")
    return resultados


if __name__ == "__main__":
    main()
