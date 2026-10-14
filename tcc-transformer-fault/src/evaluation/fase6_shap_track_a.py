import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, FIGURES_DIR, METRICS_DIR, SEEDS_AVALIACAO
from src.iga_xgboost.cromossomo import FEATURES_TRACK_A
from src.iga_xgboost.train_hybrid import LABEL_MAP_A, _carregar_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FASE6_METRICS_DIR = METRICS_DIR / "fase6"
FASE6_METRICS_DIR.mkdir(parents=True, exist_ok=True)
FASE6_FIGURES_DIR = FIGURES_DIR / "fase6"
FASE6_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df_train = _carregar_csv("katser_features_train.csv")
    df_test = _carregar_csv("katser_features_test.csv")

    X_train = df_train[FEATURES_TRACK_A].values.astype(np.float32)
    y_train = np.array([LABEL_MAP_A[v] for v in df_train["fault_type"]], dtype=np.int32)
    
    X_test = df_test[FEATURES_TRACK_A].values.astype(np.float32)
    y_test = np.array([LABEL_MAP_A[v] for v in df_test["fault_type"]], dtype=np.int32)

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, hparams, seed):
    modelo = xgb.XGBClassifier(
        **hparams,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    # XGBoost sem SMOTE (Track A usa balanceamento via hiperparâmetros ou sample weight se definido,
    # mas aqui usaremos padrão ou os hparams definidos pelo IGA)
    modelo.fit(X_train, y_train)
    return modelo


def plot_lstm_credit_migration(shap_por_seed, feature_names):
    """
    Mostra a migração de importância entre variáveis "gêmeas" (ex: H2).
    """
    if "lstm_h14_H2" not in feature_names or "lstm_h28_H2" not in feature_names:
        logger.warning("Features gêmeas de H2 não encontradas no conjunto selecionado.")
        return
        
    idx_h14 = feature_names.index("lstm_h14_H2")
    idx_h28 = feature_names.index("lstm_h28_H2")
    
    # shap_por_seed tem shape (5, n_features) (já somado pelas classes e amostras)
    impact_h14 = shap_por_seed[:, idx_h14]
    impact_h28 = shap_por_seed[:, idx_h28]
    soma = impact_h14 + impact_h28
    
    pct_h14 = (impact_h14 / soma) * 100
    pct_h28 = (impact_h28 / soma) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    seeds_str = [str(s) for s in SEEDS_AVALIACAO]
    
    # Plota a soma total em cima como referência
    ax2 = ax.twinx()
    ax2.plot(seeds_str, soma, 'o-', color='black', label='Impacto Total de H2', linewidth=2)
    ax2.set_ylabel("Soma Média de |SHAP| (H2)")
    ax2.set_ylim(0, max(soma)*1.2)
    
    # Plota barras de %
    ax.bar(seeds_str, pct_h14, label='lstm_h14_H2', color='skyblue')
    ax.bar(seeds_str, pct_h28, bottom=pct_h14, label='lstm_h28_H2', color='salmon')
    
    ax.set_ylabel("% de Crédito (SHAP)")
    ax.set_xlabel("Semente (Seed)")
    ax.set_title("Migração de Crédito SHAP entre Gêmeas (H2) por Semente")
    
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(FASE6_FIGURES_DIR / "fig7_lstm_credit_migration.png", dpi=300)
    plt.close()


def main():
    logger.info("Carregando configurações da Fase 5 (Track A)...")
    with open(METRICS_DIR / "iga_track_a_results.json", encoding="utf-8") as f:
        fase5 = json.load(f)

    c1_config = fase5["C1_IGA"]
    features_selecionadas = c1_config["features_selecionadas"]
    hparams = c1_config["hiperparametros"]

    logger.info(f"Features C1_IGA: {features_selecionadas}")
    
    X_train, y_train, X_test, y_test = load_data()
    
    idx_feats = [FEATURES_TRACK_A.index(f) for f in features_selecionadas]
    X_train_model = X_train[:, idx_feats]
    X_test_model = X_test[:, idx_feats]
    
    shap_values_seeds = []
    
    logger.info("Treinando modelos Track A para as 5 seeds e extraindo SHAP...")
    for seed in SEEDS_AVALIACAO:
        modelo = train_model(X_train_model, y_train, hparams, seed)
        booster = modelo.get_booster()
        
        dtest = xgb.DMatrix(X_test_model)
        contribs = booster.predict(dtest, pred_contribs=True)
        # Ignora bias (última coluna)
        contribs_feats = contribs[:, :, :-1]
        
        # Média de |SHAP| (por feature, combinando todas as classes)
        # Shape original = (n_samples, n_classes, n_features)
        mean_abs_por_feat = np.mean(np.abs(contribs_feats), axis=(0, 1))
        shap_values_seeds.append(mean_abs_por_feat)

    shap_values_seeds = np.array(shap_values_seeds)
    
    logger.info("Gerando Figura 7: Narrativa LSTM...")
    plot_lstm_credit_migration(shap_values_seeds, features_selecionadas)
    
    logger.info("Análise SHAP Track A concluída com sucesso.")

if __name__ == "__main__":
    main()
