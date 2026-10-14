"""
Módulo de janelas deslizantes para séries temporais.

Separado em módulo próprio para ser importado tanto pelo pré-processamento
(Fase 2) quanto pelo treinamento da LSTM (Fase 3), onde diferentes valores
de w ∈ {10, 20, 30, 50} serão testados experimentalmente.
"""

import numpy as np


def criar_janelas(
    serie: np.ndarray,
    w: int,
    h: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera pares (X, y) de janela deslizante a partir de uma única série.

    Parameters
    ----------
    serie : np.ndarray
        Array de shape (T, n_features) com T timesteps.
    w : int
        Tamanho da janela de entrada (número de timesteps).
    h : int
        Horizonte de previsão (número de passos à frente). Default = 1.

    Returns
    -------
    X : np.ndarray
        Array de shape (n_amostras, w, n_features) — janelas de entrada.
    y : np.ndarray
        Array de shape (n_amostras, h, n_features) — alvos de saída.
        Se h == 1, shape é (n_amostras, n_features).

    Raises
    ------
    ValueError
        Se a série for curta demais para gerar ao menos uma amostra.
    """
    T, n_features = serie.shape
    n_amostras = T - w - h + 1

    if n_amostras <= 0:
        raise ValueError(
            f"Série muito curta: T={T}, w={w}, h={h} → "
            f"precisaria de T >= {w + h} para gerar ao menos 1 amostra."
        )

    X = np.stack([serie[i : i + w] for i in range(n_amostras)])          # (n, w, F)
    y_raw = np.stack([serie[i + w : i + w + h] for i in range(n_amostras)])  # (n, h, F)

    # Squeeze o eixo de horizonte quando h == 1, para compatibilidade com
    # funções de perda que esperam shape (n, F).
    y = y_raw.squeeze(axis=1) if h == 1 else y_raw

    return X, y


def criar_dataset_janelado(
    series: np.ndarray,
    w: int,
    h: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica `criar_janelas` a um batch de séries e concatena os resultados.

    Parameters
    ----------
    series : np.ndarray
        Array de shape (n_series, T, n_features).
    w : int
        Tamanho da janela de entrada.
    h : int
        Horizonte de previsão. Default = 1.

    Returns
    -------
    X : np.ndarray
        Array concatenado de shape (total_amostras, w, n_features).
    y : np.ndarray
        Array concatenado de shape (total_amostras, n_features) se h==1,
        ou (total_amostras, h, n_features) se h>1.
    """
    X_list, y_list = [], []
    for serie in series:
        X_i, y_i = criar_janelas(serie, w=w, h=h)
        X_list.append(X_i)
        y_list.append(y_i)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
