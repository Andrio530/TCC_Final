"""
Fase 5 — Track B: treinamento do IGA-XGBoost na base de classificação chinesa.

Fluxo:
  1. Carrega os CSVs da Fase 4B (china_features_{train_smote,val,test}.csv)
  2. Executa o IGA para encontrar (features ótimas, hiperparâmetros ótimos)
  3. Treina o XGBoost final e avalia no teste (Experimento C3)
  4. Executa o XGBoost com parâmetros default (Experimento C4)
  5. Aplica os métodos tradicionais IEC 60599 (razões) e Duval (T1, T2 do plano)
  6. Repete treino final com 5 seeds (média ± desvio-padrão)
  7. Salva todos os resultados

Saídas (reports/metrics/ e data/processed/):
  iga_track_b_results.json    — métricas IGA, default e tradicionais
  iga_evolution_track_b.json  — histórico de fitness por geração
  iga_best_model_track_b.pkl  — modelo XGBoost final
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
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
from src.iga_xgboost.cromossomo import FEATURES_TRACK_B, decodificar  # noqa: E402
from src.iga_xgboost.iga import IGA  # noqa: E402
from src.iga_xgboost.traditional_methods import avaliar_metodos_tradicionais  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TRACK = "B"

# Mapeamento fixo de labels para 7 classes
LABEL_MAP_B: dict[str, int] = {
    "Normal": 0, "PD": 1, "D1": 2, "D2": 3, "T1": 4, "T2": 5, "T3": 6,
}
LABEL_MAP_B_INV: dict[int, str] = {v: k for k, v in LABEL_MAP_B.items()}

# Hiperparâmetros default para comparação (Experimento C4)
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


class CustomSMOTE(SMOTE):
    """SMOTE customizado para a Trilha B (9 features).
    Aplica SMOTE apenas nos 5 gases brutos e recalcula as 4 razões IEC.
    """
    def __init__(self, scaler_path, **kwargs):
        super().__init__(**kwargs)
        self.scaler_path = scaler_path
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def _fit_resample(self, X, y):
        # 1. SMOTE apenas nas 5 primeiras colunas
        X_gases = X[:, :5]
        X_res, y_res = super()._fit_resample(X_gases, y)
        
        # 2. Desfazer normalização (o scaler foi ajustado para 19 features)
        X_dummy = np.zeros((X_res.shape[0], 19))
        X_dummy[:, :5] = X_res
        
        minmax = self.scaler.named_steps["scaler"]
        # NOTA: O inverse_transform usará os min/max das 19 features.
        # As 14 últimas colunas (zeros) gerarão valores desnormalizados artificiais,
        # mas isso não é um problema pois elas são completamente sobrescritas abaixo
        # com os cálculos físicos corretos (razões e somas).
        X_raw = minmax.inverse_transform(X_dummy)
        X_raw = np.expm1(X_raw)
        
        # 3. Recalcular todas as 14 features derivadas
        # [H2, CH4, C2H6, C2H4, C2H2] = [0, 1, 2, 3, 4]
        H2, CH4, C2H6, C2H4, C2H2 = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2], X_raw[:, 3], X_raw[:, 4]
        
        def razao(n, d):
            return np.where(d > 0, n / d, 0.0)
            
        # Ratios IEC + Doernenburg + Wu
        X_raw[:, 5] = razao(CH4, H2)
        X_raw[:, 6] = razao(C2H2, C2H4)
        X_raw[:, 7] = razao(C2H4, C2H6)
        X_raw[:, 8] = razao(C2H2, CH4)
        X_raw[:, 9] = razao(C2H2, C2H6)
        
        # Somas TH e TH1
        TH = CH4 + C2H4 + C2H2
        TH1 = TH + C2H6
        X_raw[:, 10] = TH
        X_raw[:, 11] = TH1
        
        # Ratios TH
        X_raw[:, 12] = razao(CH4, TH)
        X_raw[:, 13] = razao(C2H4, TH)
        X_raw[:, 14] = razao(C2H2, TH)
        
        # Ratios TH1
        X_raw[:, 15] = razao(CH4, TH1)
        X_raw[:, 16] = razao(C2H6, TH1)
        X_raw[:, 17] = razao(C2H4, TH1)
        X_raw[:, 18] = razao(C2H2, TH1)
        
        # 4. Refazer normalização completa
        X_final = self.scaler.transform(X_raw)
        return X_final.astype(np.float32), y_res


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
    """Carrega os três splits da base chinesa.

    O treino usa a versão SMOTE (balanceada), val e teste usam os dados originais
    para avaliação sem viés de síntese.

    Returns
    -------
    X_train_sm, y_train_sm : np.ndarray
        Treino balanceado por SMOTE (3458 × 9).
    X_val, y_val : np.ndarray
        Validação original (295 × 9).
    X_test, y_test : np.ndarray
        Teste original (295 × 9).
    """
    df_train = _carregar_csv("china_features_train.csv")
    df_val   = _carregar_csv("china_features_val.csv")
    df_test  = _carregar_csv("china_features_test.csv")

    def _separar(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[FEATURES_TRACK_B].values.astype(np.float32)
        y = np.array([LABEL_MAP_B[v] for v in df["fault_type"]], dtype=np.int32)
        return X, y

    X_train, y_train = _separar(df_train)
    X_val,   y_val   = _separar(df_val)
    X_test,  y_test  = _separar(df_test)

    logger.info(
        "Track B — treino: %s | val: %s | teste: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )
    logger.info(
        "Distribuição treino orig: %s",
        {LABEL_MAP_B_INV[k]: int(v) for k, v in
         zip(*np.unique(y_train, return_counts=True))},
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# 2. Treino e avaliação do XGBoost
# ---------------------------------------------------------------------------

def _treinar_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hparams: dict,
    seed: int,
) -> XGBClassifier:
    modelo = XGBClassifier(
        **hparams,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    modelo.fit(X_train, y_train)
    return modelo


def _avaliar_modelo(
    modelo: XGBClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    classes: list[str],
) -> dict:
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


def _avaliar_multi_seed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hparams: dict,
    features_idx: list[int],
    classes: list[str],
) -> dict:
    """Treina 5 vezes com seeds diferentes e retorna média ± desvio."""
    f1_scores, acc_scores = [], []
    for seed in SEEDS_AVALIACAO:
        m = _treinar_xgboost(X_train[:, features_idx], y_train, hparams, seed)
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
# 3. Pipeline principal
# ---------------------------------------------------------------------------

def main() -> dict:
    """Executa o pipeline completo do Track B e retorna as métricas."""
    logger.info("=" * 60)
    logger.info("Fase 5 — Track B (China, 7 classes, classificador ampliado)")
    logger.info("=" * 60)

    classes = list(LABEL_MAP_B.keys())

    # 1. Carrega dados
    X_train, y_train, X_val, y_val, X_test, y_test = carregar_dados()

    scaler_path = DATA_PROCESSED / "china_feature_scaler.pkl"
    custom_smote = CustomSMOTE(scaler_path=scaler_path, random_state=RANDOM_SEED, k_neighbors=5)

    # -------------------------------------------------------------------
    # Experimento C3 — IGA-XGBoost (otimizado)
    # -------------------------------------------------------------------
    logger.info("Executando IGA (Track B)...")
    iga = IGA(
        track=TRACK, X_train=X_train, y_train=y_train, seed=RANDOM_SEED,
        smote_sampler=custom_smote
    )
    melhor_crom = iga.evoluir()

    features_ativas, hparams_iga = decodificar(melhor_crom, TRACK)
    features_idx_iga = [FEATURES_TRACK_B.index(f) for f in features_ativas]

    logger.info("Treinando modelo final (C3 — IGA)...")
    
    # SMOTE apenas no final
    X_train_sm, y_train_sm = custom_smote.fit_resample(X_train, y_train)
    
    modelo_iga = _treinar_xgboost(
        X_train_sm[:, features_idx_iga], y_train_sm, hparams_iga, RANDOM_SEED,
    )

    metricas_c3_val  = _avaliar_modelo(modelo_iga, X_val[:, features_idx_iga],
                                       y_val, classes)
    metricas_c3_test = _avaliar_modelo(modelo_iga, X_test[:, features_idx_iga],
                                       y_test, classes)
    multiseed_c3 = _avaliar_multi_seed(X_train_sm, y_train_sm, X_test, y_test,
                                       hparams_iga, features_idx_iga, classes)

    logger.info(
        "C3 (IGA) — Val F1=%.4f | Teste F1=%.4f ± %.4f",
        metricas_c3_val["f1_macro"],
        multiseed_c3["f1_macro_mean"],
        multiseed_c3["f1_macro_std"],
    )

    # -------------------------------------------------------------------
    # Experimento C4 — XGBoost default (sem IGA)
    # -------------------------------------------------------------------
    logger.info("Treinando modelo default (C4 — sem IGA)...")
    todas_features_idx = list(range(len(FEATURES_TRACK_B)))
    modelo_default = _treinar_xgboost(X_train_sm, y_train_sm, HPARAMS_DEFAULT, RANDOM_SEED)
    metricas_c4_val  = _avaliar_modelo(modelo_default, X_val, y_val, classes)
    metricas_c4_test = _avaliar_modelo(modelo_default, X_test, y_test, classes)
    multiseed_c4 = _avaliar_multi_seed(X_train_sm, y_train_sm, X_test, y_test,
                                       HPARAMS_DEFAULT, todas_features_idx, classes)

    logger.info(
        "C4 (default) — Val F1=%.4f | Teste F1=%.4f ± %.4f",
        metricas_c4_val["f1_macro"],
        multiseed_c4["f1_macro_mean"],
        multiseed_c4["f1_macro_std"],
    )

    # -------------------------------------------------------------------
    # Métodos tradicionais T1 (IEC 60599, razões) e T2 (Duval)
    # -------------------------------------------------------------------
    logger.info("Avaliando métodos tradicionais (IEC 60599 e Duval)...")
    try:
        tradicionais = avaliar_metodos_tradicionais(
            caminho_csv_teste=DATA_PROCESSED / "classification_test.csv",
            classes_interesse=classes,
        )
    except FileNotFoundError as exc:
        logger.warning("Métodos tradicionais ignorados: %s", exc)
        tradicionais = {}

    # -------------------------------------------------------------------
    # Salvar resultados
    # -------------------------------------------------------------------
    resultados = {
        "track": "B",
        "descricao": "Classificador ampliado (7 classes IEC) na base chinesa",
        "classes": classes,
        "C3_IGA": {
            "features_selecionadas": features_ativas,
            "n_features": len(features_ativas),
            "hiperparametros": hparams_iga,
            "fitness_cv": float(iga.melhor_fitness),
            "val":        metricas_c3_val,
            "test":       metricas_c3_test,
            "multiseed":  multiseed_c3,
        },
        "C4_default": {
            "features_selecionadas": FEATURES_TRACK_B,
            "n_features": len(FEATURES_TRACK_B),
            "hiperparametros": HPARAMS_DEFAULT,
            "val":        metricas_c4_val,
            "test":       metricas_c4_test,
            "multiseed":  multiseed_c4,
        },
        "T1_IEC60599": tradicionais.get("IEC60599", {}),
        "T2_Duval":    tradicionais.get("Duval", {}),
    }

    out_metrics = METRICS_DIR / "iga_track_b_results.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    logger.info("Métricas salvas em '%s'.", out_metrics)

    out_evolucao = METRICS_DIR / "iga_evolution_track_b.json"
    with open(out_evolucao, "w", encoding="utf-8") as f:
        json.dump({
            "historico_melhor": iga.historico_melhor,
            "historico_media":  iga.historico_media,
        }, f, indent=2)
    logger.info("Histórico de evolução salvo em '%s'.", out_evolucao)

    out_modelo = DATA_PROCESSED / "iga_best_model_track_b.pkl"
    with open(out_modelo, "wb") as f:
        pickle.dump({
            "modelo": modelo_iga,
            "features_ativas": features_ativas,
            "features_idx": features_idx_iga,
            "label_map": LABEL_MAP_B,
        }, f)
    logger.info("Modelo salvo em '%s'.", out_modelo)

    logger.info("=" * 60)
    logger.info("Track B concluído.")
    return resultados


if __name__ == "__main__":
    main()
