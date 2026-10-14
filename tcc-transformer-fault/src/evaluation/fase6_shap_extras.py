import json
import logging
import pickle
import sys
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, FIGURES_DIR, METRICS_DIR
from src.iga_xgboost.cromossomo import FEATURES_TRACK_B
from src.iga_xgboost.train_classification import LABEL_MAP_B
from src.evaluation.fase6_shap_track_b import load_data, train_model, unscale_features

FASE6_FIGURES_DIR = FIGURES_DIR / "fase6"
FASE6_METRICS_DIR = METRICS_DIR / "fase6"

def generate_waterfall_and_force():
    with open(METRICS_DIR / "iga_track_b_results.json", encoding="utf-8") as f:
        fase5 = json.load(f)
    c3_config = fase5["C3_IGA"]
    features = c3_config["features_selecionadas"]
    hparams = c3_config["hiperparametros"]

    X_train_sm, y_train_sm, X_test, y_test = load_data()
    idx_feats = [FEATURES_TRACK_B.index(f) for f in features]
    X_train_model = X_train_sm[:, idx_feats]
    X_test_model = X_test[:, idx_feats]
    
    modelo_42 = train_model(X_train_model, y_train_sm, hparams, 42)
    
    class_names = [k for k, v in sorted(LABEL_MAP_B.items(), key=lambda item: item[1])]
    
    X_test_unscaled = unscale_features(X_test_model, features)
    X_test_df = pd.DataFrame(X_test_unscaled, columns=features)
    
    explainer = shap.TreeExplainer(modelo_42)
    shap_values = explainer(X_test_df)
    
    preds = modelo_42.predict(X_test_model)
    proba = modelo_42.predict_proba(X_test_model)
    
    for c_idx, c_name in enumerate(class_names):
        best_idx = np.argmax(proba[:, c_idx])
        plt.figure()
        # Matplotlib renderer for force plot is deprecated in newer SHAP, but we try
        try:
            shap.plots.force(shap_values[best_idx, :, c_idx], show=False, matplotlib=True)
            plt.title(f"Force Plot: Canonical {c_name} (Prob: {proba[best_idx, c_idx]:.2f})")
            plt.tight_layout()
            plt.savefig(FASE6_FIGURES_DIR / f"fig5_force_canonical_{c_name}.png", dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Skipping force plot for {c_name} due to error: {e}")
        plt.close('all')
        
    erros_idx = np.where(preds != y_test)[0]
    
    count = 0
    for idx in erros_idx:
        true_label = class_names[y_test[idx]]
        pred_label = class_names[preds[idx]]
        
        plt.figure(figsize=(8,6))
        try:
            shap.plots.waterfall(shap_values[idx, :, preds[idx]], show=False)
            plt.title(f"Waterfall Error: True={true_label} | Pred={pred_label}")
            plt.tight_layout()
            plt.savefig(FASE6_FIGURES_DIR / f"fig6_waterfall_error_{true_label}_to_{pred_label}_{idx}.png", dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Skipping waterfall plot for idx {idx} due to error: {e}")
        plt.close('all')
        count += 1
        if count >= 3:
            break
            
def plot_duval_triangle():
    df_test = pd.read_csv(DATA_PROCESSED / "china_features_test.csv")
    
    ch4 = df_test["CH4"].values
    c2h4 = df_test["C2H4"].values
    c2h2 = df_test["C2H2"].values
    
    total = ch4 + c2h4 + c2h2
    mask = total > 0
    pch4 = np.zeros_like(ch4)
    pc2h4 = np.zeros_like(c2h4)
    pc2h2 = np.zeros_like(c2h2)
    
    pch4[mask] = 100.0 * ch4[mask] / total[mask]
    pc2h4[mask] = 100.0 * c2h4[mask] / total[mask]
    pc2h2[mask] = 100.0 * c2h2[mask] / total[mask]
    
    x = 0.5 * (2 * pc2h4 + pch4) / 100.0
    y = (np.sqrt(3)/2.0) * pch4 / 100.0
    
    with open(METRICS_DIR / "iga_track_b_results.json", encoding="utf-8") as f:
        fase5 = json.load(f)
    c3_config = fase5["C3_IGA"]
    features = c3_config["features_selecionadas"]
    hparams = c3_config["hiperparametros"]

    X_train_sm, y_train_sm, X_test, y_test = load_data()
    idx_feats = [FEATURES_TRACK_B.index(f) for f in features]
    X_train_model = X_train_sm[:, idx_feats]
    X_test_model = X_test[:, idx_feats]
    modelo_42 = train_model(X_train_model, y_train_sm, hparams, 42)
    preds = modelo_42.predict(X_test_model)
    
    plt.figure(figsize=(10, 8))
    
    # Desenhar triângulo básico
    plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=2)
    plt.text(0, -0.05, "%C2H2", ha='center', va='top', fontsize=12)
    plt.text(1, -0.05, "%C2H4", ha='center', va='top', fontsize=12)
    plt.text(0.5, np.sqrt(3)/2 + 0.02, "%CH4", ha='center', va='bottom', fontsize=12)
    
    erros_mask = (y_test != preds)
    acertos_mask = (y_test == preds)
    
    plt.scatter(x[acertos_mask & mask], y[acertos_mask & mask], c='blue', alpha=0.5, s=20, label='Acertos', marker='o')
    plt.scatter(x[erros_mask & mask], y[erros_mask & mask], c='red', alpha=0.8, s=40, label='Erros (XGBoost)', marker='x')
    
    plt.title("Projeção Triângulo de Duval (Acertos vs Erros)")
    plt.axis('equal')
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FASE6_FIGURES_DIR / "fig5_duval_triangle.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Gerando Waterfall e Force plots...")
    generate_waterfall_and_force()
    print("Gerando Triângulo de Duval...")
    plot_duval_triangle()
    print("Concluído!")
