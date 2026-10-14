"""
Reprodução do modelo IGA-XGBoost de Wu et al. (2021).
Utiliza as 18 features originais da Tabela 1 do artigo.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, METRICS_DIR, RANDOM_SEED, SEEDS_AVALIACAO
from src.iga_xgboost.cromossomo import FEATURES_TRACK_WU, decodificar
from src.iga_xgboost.iga import IGA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TRACK = "WU"

LABEL_MAP_B: dict[str, int] = {
    "Normal": 0, "PD": 1, "D1": 2, "D2": 3, "T1": 4, "T2": 5, "T3": 6,
}
LABEL_MAP_B_INV: dict[int, str] = {v: k for k, v in LABEL_MAP_B.items()}

# Parâmetros de busca reduzidos para equiparidade com o C3_IGA (TCC)
# Wu original usou 200, 200. Nós usamos 50, 100 para ser comparável.
POP_SIZE = 50
MAX_GEN = 100


def build_wu_features(df: pd.DataFrame) -> np.ndarray:
    """Constrói as 18 features de Wu et al. (2021)."""
    eps = 1e-9
    
    h2 = df["H2"].values.astype(float)
    ch4 = df["CH4"].values.astype(float)
    c2h6 = df["C2H6"].values.astype(float)
    c2h4 = df["C2H4"].values.astype(float)
    c2h2 = df["C2H2"].values.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_ch4_h2 = np.where(h2 > 0, ch4 / h2, 0.0)
        r_c2h4_c2h6 = np.where(c2h6 > 0, c2h4 / c2h6, 0.0)
        r_c2h2_c2h4 = np.where(c2h4 > 0, c2h2 / c2h4, 0.0)
        r_c2h2_c2h6 = np.where(c2h6 > 0, c2h2 / c2h6, 0.0)
        
        th = ch4 + c2h4 + c2h2
        th1 = ch4 + c2h4 + c2h2 + c2h6
        
        r_ch4_th = np.where(th > 0, ch4 / th, 0.0)
        r_c2h4_th = np.where(th > 0, c2h4 / th, 0.0)
        r_c2h2_th = np.where(th > 0, c2h2 / th, 0.0)
        
        r_ch4_th1 = np.where(th1 > 0, ch4 / th1, 0.0)
        r_c2h6_th1 = np.where(th1 > 0, c2h6 / th1, 0.0)
        r_c2h4_th1 = np.where(th1 > 0, c2h4 / th1, 0.0)
        r_c2h2_th1 = np.where(th1 > 0, c2h2 / th1, 0.0)

    # A ordem deve bater com FEATURES_TRACK_WU
    # "H2", "CH4", "C2H6", "C2H4", "C2H2",
    # "ratio_CH4_H2", "ratio_C2H4_C2H6", "ratio_C2H2_C2H4", "ratio_C2H2_C2H6",
    # "TH", "TH1",
    # "ratio_CH4_TH", "ratio_C2H4_TH", "ratio_C2H2_TH",
    # "ratio_CH4_TH1", "ratio_C2H6_TH1", "ratio_C2H4_TH1", "ratio_C2H2_TH1",
    features = np.column_stack([
        h2, ch4, c2h6, c2h4, c2h2,
        r_ch4_h2, r_c2h4_c2h6, r_c2h2_c2h4, r_c2h2_c2h6,
        th, th1,
        r_ch4_th, r_c2h4_th, r_c2h2_th,
        r_ch4_th1, r_c2h6_th1, r_c2h4_th1, r_c2h2_th1
    ])
    return features


class WuCustomSMOTE(SMOTE):
    def __init__(self, pipeline_path, **kwargs):
        super().__init__(**kwargs)
        self.pipeline_path = pipeline_path
        with open(pipeline_path, "rb") as f:
            self.pipeline = pickle.load(f)

    def _fit_resample(self, X, y):
        # SMOTE apenas nas 5 primeiras colunas (gases normalizados)
        X_gases = X[:, :5]
        X_res, y_res = super()._fit_resample(X_gases, y)
        
        # Desfazer normalização
        X_dummy = np.zeros((X_res.shape[0], 18))
        X_dummy[:, :5] = X_res
        
        # Inverter minmax e log1p (expm1)
        minmax = self.pipeline.named_steps["scaler"]
        # NOTA: O inverse_transform usará os min/max das 18 features.
        # As últimas colunas (zeros) gerarão valores desnormalizados artificiais,
        # mas isso não é um problema pois elas são completamente sobrescritas pela função
        # build_wu_features abaixo, recriando as features fisicamente consistentes.
        X_raw_dummy = minmax.inverse_transform(X_dummy)
        X_raw = np.expm1(X_raw_dummy)
        
        # Recalcular as 18 features fisicamente corretas usando pandas falso
        df_raw = pd.DataFrame(X_raw[:, :5], columns=["H2", "CH4", "C2H6", "C2H4", "C2H2"])
        X_full_raw = build_wu_features(df_raw)
        
        # Refazer normalização
        X_final = self.pipeline.transform(X_full_raw)
        return X_final.astype(np.float32), y_res


def carregar_e_preparar_dados():
    df_train = pd.read_csv(DATA_PROCESSED / "classification_train.csv")
    df_val = pd.read_csv(DATA_PROCESSED / "classification_val.csv")
    df_test = pd.read_csv(DATA_PROCESSED / "classification_test.csv")
    
    y_train = np.array([LABEL_MAP_B[v] for v in df_train["fault_type"]], dtype=np.int32)
    y_val = np.array([LABEL_MAP_B[v] for v in df_val["fault_type"]], dtype=np.int32)
    y_test = np.array([LABEL_MAP_B[v] for v in df_test["fault_type"]], dtype=np.int32)

    X_train_raw = build_wu_features(df_train)
    X_val_raw = build_wu_features(df_val)
    X_test_raw = build_wu_features(df_test)
    
    pipeline = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, validate=True)),
        ("scaler", MinMaxScaler()),
    ])
    pipeline.fit(X_train_raw)
    
    X_train = pipeline.transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)
    
    pipeline_path = DATA_PROCESSED / "wu_feature_scaler.pkl"
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)
        
    return X_train, y_train, X_val, y_val, X_test, y_test, pipeline_path


def _treinar_xgboost(X_train, y_train, hparams, seed):
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


def _avaliar_modelo(modelo, X, y_true, classes):
    y_pred = modelo.predict(X)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classes": classes,
    }

def _avaliar_multi_seed(X_train, y_train, X_test, y_test, hparams, features_idx, classes):
    f1_scores, acc_scores = [], []
    for seed in SEEDS_AVALIACAO:
        m = _treinar_xgboost(X_train[:, features_idx], y_train, hparams, seed)
        y_pred = m.predict(X_test[:, features_idx])
        f1_scores.append(float(f1_score(y_test, y_pred, average="macro", zero_division=0)))
        acc_scores.append(float(accuracy_score(y_test, y_pred)))

    return {
        "f1_macro_mean": float(np.mean(f1_scores)),
        "f1_macro_std": float(np.std(f1_scores)),
        "accuracy_mean": float(np.mean(acc_scores)),
        "accuracy_std": float(np.std(acc_scores)),
        "f1_por_seed": f1_scores,
        "seeds": SEEDS_AVALIACAO,
    }


def main():
    logger.info("=" * 60)
    logger.info("Treinando Baseline Wu et al. (2021) com 18 features")
    logger.info("=" * 60)

    classes = list(LABEL_MAP_B.keys())

    X_train, y_train, X_val, y_val, X_test, y_test, pipeline_path = carregar_e_preparar_dados()

    custom_smote = WuCustomSMOTE(pipeline_path=pipeline_path, random_state=RANDOM_SEED, k_neighbors=5)

    # IGA usa pop=50, max_gen=100 por padrão nas importações do config,
    # então usará os mesmos parâmetros de C3_IGA para comparação justa.
    logger.info("Executando IGA (Wu Baseline)...")
    iga = IGA(
        track=TRACK, X_train=X_train, y_train=y_train, seed=RANDOM_SEED,
        smote_sampler=custom_smote
    )
    melhor_crom = iga.evoluir()

    features_ativas, hparams_iga = decodificar(melhor_crom, TRACK)
    features_idx_iga = [FEATURES_TRACK_WU.index(f) for f in features_ativas]

    logger.info("Treinando modelo final...")
    X_train_sm, y_train_sm = custom_smote.fit_resample(X_train, y_train)
    
    modelo_iga = _treinar_xgboost(
        X_train_sm[:, features_idx_iga], y_train_sm, hparams_iga, RANDOM_SEED,
    )

    metricas_val = _avaliar_modelo(modelo_iga, X_val[:, features_idx_iga], y_val, classes)
    metricas_test = _avaliar_modelo(modelo_iga, X_test[:, features_idx_iga], y_test, classes)
    multiseed = _avaliar_multi_seed(X_train_sm, y_train_sm, X_test, y_test, hparams_iga, features_idx_iga, classes)

    logger.info("Wu 2021 Baseline — Val F1=%.4f | Teste F1=%.4f ± %.4f",
                metricas_val["f1_macro"], multiseed["f1_macro_mean"], multiseed["f1_macro_std"])

    # Atualizando o JSON
    out_metrics = METRICS_DIR / "iga_track_b_results.json"
    if out_metrics.exists():
        with open(out_metrics, "r", encoding="utf-8") as f:
            resultados = json.load(f)
    else:
        resultados = {}

    resultados["T3_Wu2021"] = {
        "features_selecionadas": features_ativas,
        "n_features": len(features_ativas),
        "hiperparametros": hparams_iga,
        "fitness_cv": float(iga.melhor_fitness),
        "val": metricas_val,
        "test": metricas_test,
        "multiseed": multiseed,
    }

    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    logger.info("Métricas de Wu 2021 salvas no JSON.")

if __name__ == "__main__":
    main()
