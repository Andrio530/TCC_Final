import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

METRICS_DIR = Path('reports/metrics')
FIGURES_DIR = Path('reports/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Carregar Track A e B
with open(METRICS_DIR / 'iga_track_a_results.json', 'r', encoding='utf-8') as f:
    res_a = json.load(f)
with open(METRICS_DIR / 'iga_track_b_results.json', 'r', encoding='utf-8') as f:
    res_b = json.load(f)

# Dados para o gráfico
labels = [
    'Track A\nIGA-XGBoost\n(4 Classes)', 
    'Track A\nXGBoost Padrão\n(4 Classes)', 
    'Track B\nIGA-XGBoost\n(7 Classes)', 
    'Track B\nXGBoost Padrão\n(7 Classes)', 
    'Wu et al. (2021)\nBaseline\n(7 Classes)'
]

f1_scores = [
    res_a['C1_IGA']['multiseed']['f1_macro_mean'],
    res_a['C2_default']['multiseed']['f1_macro_mean'],
    res_b['C3_IGA']['multiseed']['f1_macro_mean'],
    res_b['C4_default']['multiseed']['f1_macro_mean'],
    res_b['T3_Wu2021']['multiseed']['f1_macro_mean']
]

erros = [
    res_a['C1_IGA']['multiseed']['f1_macro_std'],
    0.0, # Default sem IGA roda uma vez só, variância = 0 ou não aplicável aqui
    res_b['C3_IGA']['multiseed']['f1_macro_std'],
    0.0,
    res_b['T3_Wu2021']['multiseed']['f1_macro_std']
]

x = np.arange(len(labels))
width = 0.6

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, f1_scores, width, yerr=erros, capsize=5, 
              color=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c'],
              edgecolor='black')

ax.set_ylabel('F1-Macro (Multi-seed)')
ax.set_title('Comparação de Desempenho (F1-Macro) - Modelos XGBoost')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionar os valores acima das barras
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'xgb_f1_comparison.png', dpi=300)
plt.close()

print("Gráfico de barras gerado com sucesso!")
