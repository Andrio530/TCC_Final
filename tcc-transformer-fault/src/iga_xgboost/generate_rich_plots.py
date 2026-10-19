import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

METRICS_DIR = Path('reports/metrics')
FIGURES_DIR = Path('reports/figures')
MODELS_DIR = Path('data/processed')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Diretório das figuras do documento LaTeX
TCC_FIGURES_DIR = Path('../tcc-unisinos/figuras')

# 1. Gráfico de Trade-off (Complexidade vs Performance)
# Valores finais auditados e consolidados nas Tabelas 5 e 7
dados_tradeoff = [
    {"Modelo": "Track A - Padrão", "F1": 0.8911, "Features": 17, "Cor": "#aec7e8", "Marker": "s", "offset": (10, 6)},
    {"Modelo": "Track A - IGA", "F1": 0.8909, "Features": 8, "Cor": "#1f77b4", "Marker": "o", "offset": (10, -6)},
    {"Modelo": "Track B - Padrão", "F1": 0.7843, "Features": 19, "Cor": "#ffbb78", "Marker": "s", "offset": (10, -12)},
    {"Modelo": "Track B - IGA", "F1": 0.7970, "Features": 16, "Cor": "#ff7f0e", "Marker": "o", "offset": (-55, 12)},
    {"Modelo": "Track B - Wu et al. (2021)", "F1": 0.7829, "Features": 14, "Cor": "#2ca02c", "Marker": "^", "offset": (-155, -12)}
]

plt.figure(figsize=(10, 6))
for d in dados_tradeoff:
    plt.scatter(d["Features"], d["F1"], color=d["Cor"], marker=d["Marker"], s=220, label=f"{d['Modelo']} ({d['Features']} feats, F1={d['F1']:.4f})", edgecolors='black', zorder=4)
    plt.annotate(
        f"{d['Modelo']}\n({d['Features']} feats)",
        (d["Features"], d["F1"]),
        xytext=d["offset"],
        textcoords='offset points',
        fontsize=9,
        fontweight='semibold',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="gray", linewidth=0.5)
    )

plt.title('Trade-off: Complexidade Dimensional vs Desempenho Diagnóstico', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Número de Atributos (Complexidade)', fontsize=11)
plt.ylabel('F1-Macro (Conjunto de Teste)', fontsize=11)
plt.xlim(6, 21)
plt.ylim(0.76, 0.92)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9.5)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'xgb_tradeoff_complexity.png', dpi=300)
if TCC_FIGURES_DIR.exists():
    plt.savefig(TCC_FIGURES_DIR / 'xgb_tradeoff_complexity.png', dpi=300)
plt.close()

# 2. Gráfico de Importância das Features (Track A)
try:
    best_a = joblib.load(MODELS_DIR / 'iga_best_model_track_a.pkl')
    # Extrair modelo do pipeline (se houver)
    modelo_xg = best_a['modelo']
    if hasattr(modelo_xg, 'named_steps'):
        modelo_xg = modelo_xg.named_steps['xgbclassifier']
    
    features = best_a['features_ativas']
    importances = modelo_xg.feature_importances_
    
    df_imp = pd.DataFrame({'Feature': features, 'Importância': importances})
    df_imp = df_imp.sort_values(by='Importância', ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_imp['Feature'], df_imp['Importância'], color='steelblue', edgecolor='black')
    plt.title('Importância das Features (Track A - IGA-XGBoost)')
    plt.xlabel('Importância Relativa (Gini)')
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'xgb_feature_importance_track_a.png', dpi=300)
    plt.close()
    print("Gerou importâncias Track A.")
except Exception as e:
    print(f"Não conseguiu gerar importância Track A: {e}")

# 3. Gráfico de Importância das Features (Track B)
try:
    best_b = joblib.load(MODELS_DIR / 'iga_best_model_track_b.pkl')
    modelo_xg = best_b['modelo']
    if hasattr(modelo_xg, 'named_steps'):
        modelo_xg = modelo_xg.named_steps['xgbclassifier']
    
    features = best_b['features_ativas']
    importances = modelo_xg.feature_importances_
    
    df_imp = pd.DataFrame({'Feature': features, 'Importância': importances})
    df_imp = df_imp.sort_values(by='Importância', ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_imp['Feature'], df_imp['Importância'], color='darkorange', edgecolor='black')
    plt.title('Importância das Features (Track B - IGA-XGBoost)')
    plt.xlabel('Importância Relativa (Gini)')
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'xgb_feature_importance_track_b.png', dpi=300)
    plt.close()
    print("Gerou importâncias Track B.")
except Exception as e:
    print(f"Não conseguiu gerar importância Track B: {e}")
