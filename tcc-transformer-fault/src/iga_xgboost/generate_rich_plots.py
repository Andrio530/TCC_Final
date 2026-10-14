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

# 1. Gráfico de Trade-off (Complexidade vs Performance)
with open(METRICS_DIR / 'iga_track_a_results.json', 'r', encoding='utf-8') as f:
    res_a = json.load(f)
with open(METRICS_DIR / 'iga_track_b_results.json', 'r', encoding='utf-8') as f:
    res_b = json.load(f)

dados_tradeoff = [
    {"Modelo": "Track A - Default", "F1": res_a['C2_default']['test']['f1_macro'], "Features": 15, "Cor": "#aec7e8", "Marker": "s"},
    {"Modelo": "Track A - IGA", "F1": res_a['C1_IGA']['test']['f1_macro'], "Features": res_a['C1_IGA']['n_features'], "Cor": "#1f77b4", "Marker": "o"},
    {"Modelo": "Track B - Default", "F1": res_b['C4_default']['test']['f1_macro'], "Features": 9, "Cor": "#ffbb78", "Marker": "s"},
    {"Modelo": "Track B - IGA", "F1": res_b['C3_IGA']['test']['f1_macro'], "Features": res_b['C3_IGA']['n_features'], "Cor": "#ff7f0e", "Marker": "o"},
    {"Modelo": "Track B - Wu2021", "F1": res_b['T3_Wu2021']['test']['f1_macro'], "Features": 18, "Cor": "#2ca02c", "Marker": "^"}
]

plt.figure(figsize=(10, 6))
for d in dados_tradeoff:
    plt.scatter(d["Features"], d["F1"], color=d["Cor"], marker=d["Marker"], s=200, label=d["Modelo"], edgecolors='black')
    plt.annotate(d["Modelo"].split(' - ')[-1], (d["Features"], d["F1"]), xytext=(10, -5), textcoords='offset points')

plt.title('Trade-off: Complexidade vs Desempenho')
plt.xlabel('Número de Features (Complexidade)')
plt.ylabel('F1-Macro (Teste)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'xgb_tradeoff_complexity.png', dpi=300)
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
