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
            
def _bary_to_cart(pch4, pc2h4, pc2h2):
    x = 0.5 * (2.0 * np.asarray(pc2h4) + np.asarray(pch4)) / 100.0
    y = (np.sqrt(3.0) / 2.0) * np.asarray(pch4) / 100.0
    return x, y


def plot_duval_triangle():
    # Lê as concentrações brutas em PPM (classification_test.csv)
    df_test_raw = pd.read_csv(DATA_PROCESSED / "classification_test.csv")
    
    ch4 = df_test_raw["CH4"].values.astype(float)
    c2h4 = df_test_raw["C2H4"].values.astype(float)
    c2h2 = df_test_raw["C2H2"].values.astype(float)
    
    total = ch4 + c2h4 + c2h2
    mask = total > 0
    pch4 = np.zeros_like(ch4)
    pc2h4 = np.zeros_like(c2h4)
    pc2h2 = np.zeros_like(c2h2)
    
    pch4[mask] = 100.0 * ch4[mask] / total[mask]
    pc2h4[mask] = 100.0 * c2h4[mask] / total[mask]
    pc2h2[mask] = 100.0 * c2h2[mask] / total[mask]
    
    x, y = _bary_to_cart(pch4, pc2h4, pc2h2)
    
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
    
    fig, ax = plt.subplots(figsize=(10, 8.5))
    
    # Polígonos das zonas do Triângulo 1 de Duval (em coordenadas (%CH4, %C2H4, %C2H2))
    # Cores bem distintas entre si para destacar visualmente cada região (especialmente PD, T1, T2 e T3)
    zonas_duval = [
        ("PD", [(100, 0, 0), (98, 2, 0), (98, 0, 2)], "#4dabf7", 0.75),
        ("T1", [(98, 0, 2), (98, 2, 0), (80, 20, 0), (76, 20, 4), (96, 0, 4)], "#ffe066", 0.65),
        ("T2", [(80, 20, 0), (50, 50, 0), (46, 50, 4), (76, 20, 4)], "#ff922b", 0.58),
        ("T3", [(50, 50, 0), (0, 100, 0), (0, 85, 15), (35, 50, 15)], "#ff6b6b", 0.45),
        ("D1", [(87, 0, 13), (64, 23, 13), (0, 23, 77), (0, 0, 100)], "#b197fc", 0.42),
        ("D2", [(64, 23, 13), (47, 40, 13), (31, 40, 29), (0, 71, 29), (0, 23, 77)], "#f783ac", 0.42),
        ("DT", [(96, 0, 4), (46, 50, 4), (35, 50, 15), (0, 85, 15), (0, 71, 29), (31, 40, 29), (47, 40, 13), (87, 0, 13)], "#8ce99a", 0.45),
    ]
    
    cor_mapa = {}
    for nome_z, pts_bary, cor_z, alfa_z in zonas_duval:
        cor_mapa[nome_z] = cor_z
        bx = [p[0] for p in pts_bary] + [pts_bary[0][0]]
        by = [p[1] for p in pts_bary] + [pts_bary[0][1]]
        bz = [p[2] for p in pts_bary] + [pts_bary[0][2]]
        cx, cy = _bary_to_cart(bx, by, bz)
        ax.fill(cx[:-1], cy[:-1], facecolor=cor_z, alpha=alfa_z, edgecolor="none", zorder=1)
        ax.plot(cx, cy, color="#2b2b2b", linewidth=1.4, linestyle="--", alpha=0.90, zorder=5)
    
    # Destaque sólido para as fronteiras estreitas entre PD/T1, T1/T2 e T2/T3 na borda direita
    fronteiras_termicas = [
        [(98, 2, 0), (98, 0, 2)],   # PD / T1
        [(80, 20, 0), (76, 20, 4)], # T1 / T2
        [(50, 50, 0), (46, 50, 4)], # T2 / T3
    ]
    for seg in fronteiras_termicas:
        sx, sy = _bary_to_cart([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], [seg[0][2], seg[1][2]])
        ax.plot(sx, sy, color="#111111", linewidth=2.2, linestyle="-", zorder=6)
    
    # Contorno externo do triângulo
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=2.4, zorder=6)
    
    # Rótulos das zonas com fundo colorido correspondente a cada região
    rotulos_zonas = [
        ("PD", (98.8, 0.6, 0.6), (0.61, np.sqrt(3)/2 - 0.005)),
        ("T1", (89, 9, 2), (0.68, 0.73)),
        ("T2", (65, 33, 2), (0.81, 0.52)),
        ("T3", (18, 74, 8), None),
        ("DT", (56, 34, 10), None),
        ("D1", (35, 10, 55), None),
        ("D2", (10, 33, 57), None),
    ]
    for nome_z, bary_c, pos_custom in rotulos_zonas:
        zx, zy = _bary_to_cart(*bary_c)
        cor_box = cor_mapa[nome_z]
        if pos_custom is not None:
            ax.annotate(
                nome_z, xy=(zx, zy), xytext=pos_custom,
                fontsize=11, fontweight="bold", color="#111111",
                arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.4, mutation_scale=12),
                bbox=dict(boxstyle="round,pad=0.28", facecolor=cor_box, alpha=0.92, edgecolor="#222222", linewidth=1.2),
                zorder=8
            )
        else:
            ax.text(
                zx, zy, nome_z, fontsize=11, fontweight="bold", color="#111111",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.28", facecolor=cor_box, alpha=0.90, edgecolor="#222222", linewidth=1.2),
                zorder=8
            )
    
    # Rótulos dos vértices/eixos
    ax.text(0.0, -0.04, "100% $\\mathrm{C_2H_2}$\n(0% $\\mathrm{C_2H_4}$)", ha='center', va='top', fontsize=11.5, fontweight='bold')
    ax.text(1.0, -0.04, "100% $\\mathrm{C_2H_4}$\n(0% $\\mathrm{CH_4}$)", ha='center', va='top', fontsize=11.5, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.025, "100% $\\mathrm{CH_4}$\n(0% $\\mathrm{C_2H_2}$)", ha='center', va='bottom', fontsize=11.5, fontweight='bold')
    
    erros_mask = (y_test != preds)
    acertos_mask = (y_test == preds)
    
    n_acertos = int(np.sum(acertos_mask & mask))
    n_erros = int(np.sum(erros_mask & mask))
    
    ax.scatter(x[acertos_mask & mask], y[acertos_mask & mask], c='#1f77b4', alpha=0.65, s=34, label=f'Acertos XGBoost (n={n_acertos})', marker='o', edgecolors='white', linewidth=0.4, zorder=4)
    ax.scatter(x[erros_mask & mask], y[erros_mask & mask], c='#d62728', alpha=0.90, s=58, label=f'Erros XGBoost (n={n_erros})', marker='x', linewidths=1.8, zorder=6)
    
    ax.set_title("Projeção no Triângulo 1 de Duval — Acertos vs. Erros (Conjunto de Teste, PPM Bruto)", fontsize=13, fontweight="bold", pad=38)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', frameon=True, fontsize=11, facecolor='white', framealpha=0.95, edgecolor='gray')
    plt.tight_layout()
    out_fig = FASE6_FIGURES_DIR / "fig5_duval_triangle.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    tcc_fig = ROOT.parent / "tcc-unisinos" / "figuras" / "fase6" / "fig5_duval_triangle.png"
    if tcc_fig.parent.exists():
        plt.savefig(tcc_fig, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Gerando Waterfall e Force plots...")
    generate_waterfall_and_force()
    print("Gerando Triângulo de Duval...")
    plot_duval_triangle()
    print("Concluído!")
