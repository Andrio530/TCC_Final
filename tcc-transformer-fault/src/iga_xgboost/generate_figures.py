import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

METRICS_DIR = Path('reports/metrics')
FIGURES_DIR = Path('reports/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

def plot_convergence(hist_a, hist_b, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(hist_a)+1), hist_a, label='Track A (Katser)', color='blue', marker='o', markersize=4)
    plt.plot(range(1, len(hist_b)+1), hist_b, label='Track B (China)', color='red', marker='s', markersize=4)
    plt.title('Convergência do IGA (Evolução do Fitness)')
    plt.xlabel('Geração')
    plt.ylabel('Fitness (F1-macro no Treino)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

# Carregar Track A
with open(METRICS_DIR / 'iga_track_a_results.json', 'r', encoding='utf-8') as f:
    res_a = json.load(f)

# Carregar Track B
with open(METRICS_DIR / 'iga_track_b_results.json', 'r', encoding='utf-8') as f:
    res_b = json.load(f)

# Matrizes de confusão
cm_a = res_a['C1_IGA']['test']['confusion_matrix']
classes_a = res_a['classes']
plot_confusion_matrix(cm_a, classes_a, 'Matriz de Confusão (Teste) - Track A (IGA-XGBoost)', 'xgb_confusion_track_a.png')

cm_b = res_b['C3_IGA']['test']['confusion_matrix']
classes_b = res_b['classes']
plot_confusion_matrix(cm_b, classes_b, 'Matriz de Confusão (Teste) - Track B (IGA-XGBoost)', 'xgb_confusion_track_b.png')

cm_wu = res_b['T3_Wu2021']['test']['confusion_matrix']
plot_confusion_matrix(cm_wu, classes_b, 'Matriz de Confusão (Teste) - Baseline Wu et al. 2021', 'xgb_confusion_wu2021.png')

# Convergência
with open(METRICS_DIR / 'iga_evolution_track_a.json', 'r', encoding='utf-8') as f:
    evo_a = json.load(f)['historico_melhor']

with open(METRICS_DIR / 'iga_evolution_track_b.json', 'r', encoding='utf-8') as f:
    evo_b = json.load(f)['historico_melhor']

plot_convergence(evo_a, evo_b, 'iga_convergence.png')

print("Figuras do XGBoost e IGA geradas com sucesso na pasta reports/figures/")
