import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    lowess = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, FIGURES_DIR, METRICS_DIR, RANDOM_SEED, SEEDS_AVALIACAO
from src.iga_xgboost.cromossomo import FEATURES_TRACK_B
from src.iga_xgboost.train_classification import CustomSMOTE, LABEL_MAP_B, _carregar_csv

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
    """Carrega treino e teste, aplica SMOTE no treino e extrai as features do IGA."""
    df_train = _carregar_csv("china_features_train.csv")
    df_test = _carregar_csv("china_features_test.csv")

    X_train_raw = df_train[FEATURES_TRACK_B].values.astype(np.float32)
    y_train_raw = np.array([LABEL_MAP_B[v] for v in df_train["fault_type"]], dtype=np.int32)
    
    X_test_raw = df_test[FEATURES_TRACK_B].values.astype(np.float32)
    y_test_raw = np.array([LABEL_MAP_B[v] for v in df_test["fault_type"]], dtype=np.int32)

    logger.info("Aplicando SMOTE no treino...")
    smote_c = CustomSMOTE(scaler_path=DATA_PROCESSED / "china_feature_scaler.pkl",
                          random_state=RANDOM_SEED, k_neighbors=5)
    X_train_sm, y_train_sm = smote_c.fit_resample(X_train_raw, y_train_raw)

    return X_train_sm, y_train_sm, X_test_raw, y_test_raw


def unscale_features(X_scaled, feature_names):
    """
    Usa o scaler (que esperava as 19 features originais) para desnormalizar as features.
    Injeta zeros nas features ausentes, aplica o inverse_transform e expm1.
    """
    scaler_path = DATA_PROCESSED / "china_feature_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    X_dummy = np.zeros((X_scaled.shape[0], len(FEATURES_TRACK_B)), dtype=np.float32)
    
    idx_map = [FEATURES_TRACK_B.index(f) for f in feature_names]
    for i, idx in enumerate(idx_map):
        X_dummy[:, idx] = X_scaled[:, i]
        
    minmax = scaler.named_steps["scaler"]
    X_inv_log = minmax.inverse_transform(X_dummy)
    X_raw = np.expm1(X_inv_log)
    
    # Extrai de volta apenas as colunas de interesse
    X_out = np.zeros_like(X_scaled)
    for i, idx in enumerate(idx_map):
        X_out[:, i] = X_raw[:, idx]
        
    return X_out


def train_model(X_train, y_train, hparams, seed):
    modelo = xgb.XGBClassifier(
        **hparams,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    modelo.fit(X_train, y_train)
    return modelo


def plot_heatmap(mean_abs_shap, feature_names, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_abs_shap, aspect="auto", cmap="Reds")
    plt.colorbar(label="Média de |SHAP| (impacto global)")
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xticks(range(len(class_names)), class_names)
    plt.title("Importância Global por Classe (Heatmap)")
    plt.tight_layout()
    plt.savefig(FASE6_FIGURES_DIR / "fig1_heatmap_track_b.png", dpi=300)
    plt.close()


def plot_grouped_shap(mean_abs_shap, feature_names, class_names):
    familias = {
        "Gases puros": ["H2", "CH4", "C2H6", "C2H4", "C2H2"],
        "Razoes IEC/Duval": ["ratio_CH4_H2", "ratio_C2H2_C2H4", "ratio_C2H4_C2H6", "ratio_C2H2_C2H6", "ratio_C2H2_CH4"],
        "Familia Duval (TH)": ["TH", "ratio_CH4_TH", "ratio_C2H4_TH", "ratio_C2H2_TH"],
        "Familia TH1": ["TH1", "ratio_C2H6_TH1", "ratio_C2H4_TH1", "ratio_C2H2_TH1", "ratio_CH4_TH1"]
    }

    importancias_agrupadas = {f: np.zeros(len(class_names)) for f in familias.keys()}
    
    for i, feat in enumerate(feature_names):
        for familia, lista_feats in familias.items():
            if feat in lista_feats:
                importancias_agrupadas[familia] += mean_abs_shap[i, :]
                break
                
    df_plot = pd.DataFrame(importancias_agrupadas, index=class_names)
    
    df_plot.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
    plt.title("Importância Agrupada por Família de Features")
    plt.ylabel("Soma da Média de |SHAP|")
    plt.xticks(rotation=0)
    plt.legend(title="Famílias", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FASE6_FIGURES_DIR / "fig2_grouped_track_b.png", dpi=300)
    plt.close()


def extract_thresholds(X_unscaled, pred_interactions, feature_names, class_names):
    """
    Usa o SHAP main effect (a diagonal da matriz de interações) para encontrar limiares empíricos.
    """
    thresholds = {}
    
    # Cruzamentos previstos na Fase 6
    pares_interesse = [
        ("ratio_C2H2_TH", "D1"), ("ratio_C2H2_TH", "D2"),
        ("ratio_C2H4_TH", "D1"), ("ratio_C2H2_C2H4", "D1"),
        ("ratio_CH4_H2", "PD"), ("ratio_C2H4_C2H6", "T3"),
        ("ratio_C2H4_TH", "T1"), ("ratio_C2H4_TH", "T2")
    ]
    
    for feat_name, class_name in pares_interesse:
        if feat_name not in feature_names:
            continue
            
        f_idx = feature_names.index(feat_name)
        c_idx = class_names.index(class_name)
        
        # Main effect (x_i na diagonal)
        shap_main = pred_interactions[:, c_idx, f_idx, f_idx]
        x_vals = X_unscaled[:, f_idx]
        
        # Filtra divisões por zero (que geram valores x=0 nas razões)
        mask = (x_vals > 0)
        if mask.sum() < 10:
            continue
            
        x_vals_f = x_vals[mask]
        shap_f = shap_main[mask]
        
        # Para salvar a curva e reportar
        plt.figure(figsize=(6,4))
        plt.scatter(x_vals_f, shap_f, alpha=0.5, s=10)
        
        if lowess is not None and len(x_vals_f) > 5:
            smoothed = lowess(shap_f, x_vals_f, frac=0.3)
            plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2, label="LOWESS")
            
        plt.axhline(0, color="k", linestyle="--")
        
        if x_vals_f.max() > 100:
            plt.xlim(0, min(30, x_vals_f.max()))
            
        plt.title(f"SHAP Main Effect: {feat_name} -> {class_name}")
        plt.xlabel(feat_name)
        plt.ylabel("SHAP Value")
        if lowess is not None:
            plt.legend()
        plt.tight_layout()
        plt.savefig(FASE6_FIGURES_DIR / f"fig4_thresh_{feat_name}_{class_name}.png", dpi=300)
        plt.close()

    return thresholds


def main():
    logger.info("Carregando configurações da Fase 5 (Track B)...")
    with open(METRICS_DIR / "iga_track_b_results.json", encoding="utf-8") as f:
        fase5 = json.load(f)

    c3_config = fase5["C3_IGA"]
    features_selecionadas = c3_config["features_selecionadas"]
    hparams = c3_config["hiperparametros"]

    logger.info(f"Features: {features_selecionadas}")
    
    X_train_sm, y_train_sm, X_test, y_test = load_data()
    
    idx_feats = [FEATURES_TRACK_B.index(f) for f in features_selecionadas]
    X_train_model = X_train_sm[:, idx_feats]
    X_test_model = X_test[:, idx_feats]
    
    class_names = [k for k, v in sorted(LABEL_MAP_B.items(), key=lambda item: item[1])]
    
    # --- Passo 1: Estabilidade do SHAP ---
    shap_values_seeds = []
    logger.info("Treinando modelos para as 5 seeds e extraindo SHAP...")
    for seed in SEEDS_AVALIACAO:
        modelo = train_model(X_train_model, y_train_sm, hparams, seed)
        booster = modelo.get_booster()
        
        dtest = xgb.DMatrix(X_test_model)
        # SHAP (n_samples, n_classes, n_features+1)
        contribs = booster.predict(dtest, pred_contribs=True)
        margin = booster.predict(dtest, output_margin=True)
        
        assert np.allclose(contribs.sum(axis=-1), margin, atol=1e-4), "Falha na verificação de aditividade do SHAP!"
        
        # Ignora o bias (última coluna)
        contribs_feats = contribs[:, :, :-1]
        shap_values_seeds.append(contribs_feats)

    # Média absoluta por feature para cada seed: shape = (n_seeds, n_features)
    # Somamos a importância através das classes primeiro
    mean_abs_por_seed = [np.mean(np.abs(sv), axis=(0, 1)) for sv in shap_values_seeds]
    
    rho_matrix = np.zeros((len(SEEDS_AVALIACAO), len(SEEDS_AVALIACAO)))
    for i in range(len(SEEDS_AVALIACAO)):
        for j in range(len(SEEDS_AVALIACAO)):
            rho, _ = spearmanr(mean_abs_por_seed[i], mean_abs_por_seed[j])
            rho_matrix[i, j] = rho
            
    rho_mean_real = np.mean(rho_matrix[np.triu_indices(len(SEEDS_AVALIACAO), k=1)])
    logger.info(f"Média do Spearman Rho entre seeds: {rho_mean_real:.3f}")
    
    # Salvar resultados
    with open(FASE6_METRICS_DIR / "shap_stability.json", "w") as f:
        json.dump({"spearman_rho_mean": float(rho_mean_real)}, f)

    # --- Usaremos o modelo da Seed 42 para as análises principais ---
    idx_42 = SEEDS_AVALIACAO.index(42)
    shap_values_42 = shap_values_seeds[idx_42]
    
    # Recalcula as interações para a seed 42
    modelo_42 = train_model(X_train_model, y_train_sm, hparams, 42)
    booster_42 = modelo_42.get_booster()
    
    dtest = xgb.DMatrix(X_test_model)
    logger.info("Calculando interações (pred_interactions=True)...")
    # Shape: (n_samples, n_classes, n_features+1, n_features+1)
    interactions = booster_42.predict(dtest, pred_interactions=True)

    # Desnormalizar as features do teste
    X_test_unscaled = unscale_features(X_test_model, features_selecionadas)
    
    # Calcula a média do SHAP absoluto: (n_features, n_classes)
    mean_abs_shap = np.mean(np.abs(shap_values_42), axis=0).T  # Transpõe para (features, classes)
    
    # --- Passo 2: Heatmap ---
    plot_heatmap(mean_abs_shap, features_selecionadas, class_names)
    
    # --- Passo 3: Barplot Agrupado ---
    plot_grouped_shap(mean_abs_shap, features_selecionadas, class_names)
    
    # --- Passo 4: Beeswarm Plots ---
    # Convertemos o array de teste para dataframe pandas para o SHAP usar os nomes reais
    X_test_df = pd.DataFrame(X_test_unscaled, columns=features_selecionadas)
    
    for cls_name in class_names:
        c_idx = class_names.index(cls_name)
        plt.figure(figsize=(8, 6))
        # Seleciona o SHAP da classe c_idx (n_samples, n_features)
        shap_cls = shap_values_42[:, c_idx, :]
        shap.summary_plot(shap_cls, X_test_df, show=False)
        plt.title(f"SHAP Summary: Classe {cls_name}")
        plt.tight_layout()
        plt.savefig(FASE6_FIGURES_DIR / f"fig3_beeswarm_{cls_name}.png", dpi=300)
        plt.close()
        
    # --- Passo 5: Extração de limiares empíricos (Main Effect) ---
    logger.info("Extraindo limiares empíricos...")
    extract_thresholds(X_test_unscaled, interactions, features_selecionadas, class_names)
    
    # --- Passo 6: Triângulo de Duval (Opcional simplificado) ---
    # Vamos gerar um CSV simples de acertos/erros para depois ser plotado no Duval
    preds = modelo_42.predict(X_test_model)
    erros_idx = np.where(preds != y_test)[0]
    acertos_idx = np.where(preds == y_test)[0]
    
    logger.info(f"Seed 42: {len(acertos_idx)} acertos, {len(erros_idx)} erros.")
    df_erros = pd.DataFrame({
        "True_Label": [class_names[y_test[i]] for i in erros_idx],
        "Pred_Label": [class_names[preds[i]] for i in erros_idx]
    })
    for i, feat in enumerate(features_selecionadas):
        df_erros[feat] = X_test_unscaled[erros_idx, i]
        
    df_erros.to_csv(FASE6_METRICS_DIR / "erros_test_seed42.csv", index=False)
    
    logger.info("Análise SHAP Track B concluída com sucesso.")

if __name__ == "__main__":
    main()
