"""
Avaliação e visualização dos resultados da LSTM (Fase 3).

Funções:
  - calcular_metricas(): MAE, RMSE, R² por gás e agregado.
  - plotar_curvas_treinamento(): loss treino vs. validação por época.
  - plotar_comparacao_janelas(): MAE/RMSE por w, para seleção de w*.
  - plotar_previsoes_amostra(): real vs. previsto em séries de exemplo.
  - plotar_distribuicao_erros(): histograma de resíduos por gás.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, KATSER_GASES  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Métricas
# ---------------------------------------------------------------------------

def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nomes_gases: list[str] = KATSER_GASES,
) -> dict:
    """Calcula MAE, RMSE e R² por gás e agregado.

    Parameters
    ----------
    y_true : np.ndarray
        Valores reais, shape (n_amostras, n_gases).
    y_pred : np.ndarray
        Valores previstos, shape (n_amostras, n_gases).
    nomes_gases : list[str]
        Nomes dos gases, na mesma ordem das colunas.

    Returns
    -------
    dict com keys:
        'mae_por_gas', 'rmse_por_gas', 'r2_por_gas',
        'mae_agregado', 'rmse_agregado'
    """
    erros = y_true - y_pred  # (n_amostras, n_gases)

    mae_por_gas  = {g: float(np.mean(np.abs(erros[:, i])))
                    for i, g in enumerate(nomes_gases)}
    rmse_por_gas = {g: float(np.sqrt(np.mean(erros[:, i] ** 2)))
                    for i, g in enumerate(nomes_gases)}

    # R² por gás (coeficiente de determinação)
    r2_por_gas: dict[str, float] = {}
    for i, g in enumerate(nomes_gases):
        ss_res = np.sum(erros[:, i] ** 2)
        ss_tot = np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2)
        r2_por_gas[g] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Agregado: média sobre todos os gases e amostras
    mae_agregado  = float(np.mean(np.abs(erros)))
    rmse_agregado = float(np.sqrt(np.mean(erros ** 2)))

    return {
        "mae_por_gas":   mae_por_gas,
        "rmse_por_gas":  rmse_por_gas,
        "r2_por_gas":    r2_por_gas,
        "mae_agregado":  mae_agregado,
        "rmse_agregado": rmse_agregado,
    }


# ---------------------------------------------------------------------------
# 2. Curvas de treinamento
# ---------------------------------------------------------------------------

def plotar_curvas_treinamento(
    history,
    w: int,
    seed: int,
    destino: Optional[Path] = None,
) -> Path:
    """Plota loss (MSE) de treino e validação ao longo das épocas.

    Parameters
    ----------
    history : tf.keras.callbacks.History
    w : int
        Janela usada (para nomear o arquivo).
    seed : int
        Seed usada (para nomear o arquivo).
    destino : Path | None
        Caminho de saída. Default: reports/figures/lstm_loss_w{w}_seed{seed}.png

    Returns
    -------
    Path do arquivo salvo.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if destino is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        destino = FIGURES_DIR / f"lstm_loss_w{w}_seed{seed}.png"

    epochs = range(1, len(history.history["loss"]) + 1)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history.history["loss"],     label="Treino",    color="#2196F3")
    ax.plot(epochs, history.history["val_loss"], label="Validação", color="#FF5722")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8,
               label=f"Melhor época ({best_epoch})")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Curvas de Treinamento — LSTM (w={w}, seed={seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    logger.info("Curva de treinamento salva em %s", destino)
    return destino


# ---------------------------------------------------------------------------
# 3. Comparação de janelas
# ---------------------------------------------------------------------------

def plotar_comparacao_janelas(
    resultados_janelas: dict,
    destino: Optional[Path] = None,
) -> Path:
    """Gráfico de barras agrupadas: MAE e RMSE por w e por gás.

    Parameters
    ----------
    resultados_janelas : dict
        Saída de buscar_melhor_janela() — keyed by str(w).
    destino : Path | None
        Default: reports/figures/lstm_window_comparison.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if destino is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        destino = FIGURES_DIR / "lstm_window_comparison.png"

    janelas = sorted(resultados_janelas.keys(), key=int)
    gases   = list(next(iter(resultados_janelas.values()))["mae_por_gas"].keys())

    x = np.arange(len(janelas))
    width = 0.8 / len(gases)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metrica, titulo in zip(axes, ["mae_por_gas", "rmse_por_gas"],
                                   ["MAE por Janela (Validação)", "RMSE por Janela (Validação)"]):
        for j, (gas, cor) in enumerate(zip(gases, colors)):
            valores = [resultados_janelas[w][metrica][gas] for w in janelas]
            ax.bar(x + j * width - (len(gases) - 1) * width / 2,
                   valores, width, label=gas, color=cor, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"w={w}" for w in janelas])
        ax.set_title(titulo)
        ax.set_ylabel(metrica.split("_")[0].upper())
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Comparação de Tamanhos de Janela — LSTM")
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    logger.info("Gráfico de comparação de janelas salvo em %s", destino)
    return destino


# ---------------------------------------------------------------------------
# 4. Previsões de exemplo
# ---------------------------------------------------------------------------

def plotar_previsoes_amostra(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nomes_gases: list[str] = KATSER_GASES,
    n_series: int = 3,
    destino: Optional[Path] = None,
    seed: int = 42,
) -> Path:
    """Plota real vs. previsto para n_series amostras aleatórias do teste.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Shape (n_amostras, n_gases).
    n_series : int
        Número de amostras a visualizar.
    destino : Path | None
        Default: reports/figures/lstm_predictions_sample.png
    seed : int
        Seed para selecionar as amostras aleatórias.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if destino is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        destino = FIGURES_DIR / "lstm_predictions_sample.png"

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y_true), size=n_series, replace=False)

    n_gases = len(nomes_gases)
    fig, axes = plt.subplots(n_series, n_gases, figsize=(4 * n_gases, 3 * n_series))
    # Garantir 2D mesmo com n_series=1
    if n_series == 1:
        axes = axes[np.newaxis, :]

    colors_real = "#2196F3"
    colors_pred = "#FF5722"

    for row, idx in enumerate(indices):
        for col, gas in enumerate(nomes_gases):
            ax = axes[row, col]
            # Cada ponto é uma previsão de 1 passo — mostramos os primeiros 100 pontos
            show_n = min(100, len(y_true))
            ax.plot(y_true[:show_n, col], label="Real",    color=colors_real, linewidth=0.8)
            ax.plot(y_pred[:show_n, col], label="Previsto", color=colors_pred, linewidth=0.8,
                    linestyle="--")
            if row == 0:
                ax.set_title(gas)
            if col == 0:
                ax.set_ylabel(f"Amostra {idx}")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Previsões vs. Real — LSTM (primeiros 100 pontos do teste)")
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    logger.info("Gráfico de previsões salvo em %s", destino)
    return destino


# ---------------------------------------------------------------------------
# 5. Distribuição de erros
# ---------------------------------------------------------------------------

def plotar_distribuicao_erros(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nomes_gases: list[str] = KATSER_GASES,
    destino: Optional[Path] = None,
) -> Path:
    """Histograma de resíduos (erros) por gás.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Shape (n_amostras, n_gases).
    destino : Path | None
        Default: reports/figures/lstm_error_distribution.png
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if destino is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        destino = FIGURES_DIR / "lstm_error_distribution.png"

    erros = y_true - y_pred
    n_gases = len(nomes_gases)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    fig, axes = plt.subplots(1, n_gases, figsize=(4 * n_gases, 4))
    if n_gases == 1:
        axes = [axes]

    for ax, gas, cor in zip(axes, nomes_gases, colors):
        ax.hist(erros[:, nomes_gases.index(gas)], bins=60, color=cor, alpha=0.8,
                edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        mae = float(np.mean(np.abs(erros[:, nomes_gases.index(gas)])))
        ax.set_title(f"{gas}\nMAE={mae:.6f}")
        ax.set_xlabel("Erro (real − previsto)")
        ax.set_ylabel("Frequência")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Distribuição dos Erros de Previsão — LSTM")
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    logger.info("Histograma de erros salvo em %s", destino)
    return destino
