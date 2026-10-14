"""
Carregamento e preparação dos dados Katser para treinamento da LSTM (Fase 3).

Lê os arquivos .npz gerados na Fase 2 (data/processed/) e aplica o
janelamento deslizante parametrizado para produzir tensores (X, y) prontos
para model.fit().

Opção de normalização per-gas (MinMaxScaler fitado apenas no treino):
  - Os dados Katser já estão em escala pré-normalizada (faixa ~0–0.06),
    mas os 4 gases têm escalas relativas distintas (CO: max ~0.057;
    C2H2: max ~0.0007 — razão ~80x).
  - Se normalizar=True, aplica MinMaxScaler por gás para igualar as escalas.
    O scaler é fitado APENAS no treino e transformado em val/teste.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, KATSER_GASES, LSTM_HORIZONTE  # noqa: E402
from src.preprocessing.windowing import criar_dataset_janelado  # noqa: E402

logger = logging.getLogger(__name__)

N_GASES = len(KATSER_GASES)  # 4


# ---------------------------------------------------------------------------
# 1. Carga dos splits .npz
# ---------------------------------------------------------------------------

def carregar_splits() -> tuple[np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
    """Carrega os arrays .npz da Fase 2.

    Returns
    -------
    s_treino, l_treino, s_val, l_val, s_teste, l_teste : np.ndarray
        Séries (n_series, 420, 4) e labels (n_series,) para cada split.
    """
    def _load(nome: str) -> tuple[np.ndarray, np.ndarray]:
        caminho = DATA_PROCESSED / nome
        if not caminho.exists():
            raise FileNotFoundError(
                f"Arquivo {caminho} não encontrado. "
                "Execute src/preprocessing/run_preprocessing.py primeiro."
            )
        d = np.load(caminho)
        return d["series"].astype(np.float32), d["labels"]

    s_treino, l_treino = _load("katser_train.npz")
    s_val,    l_val    = _load("katser_val.npz")
    s_teste,  l_teste  = _load("katser_test.npz")

    logger.info(
        "Dados carregados — treino: %s | val: %s | teste: %s",
        s_treino.shape, s_val.shape, s_teste.shape,
    )
    return s_treino, l_treino, s_val, l_val, s_teste, l_teste


# ---------------------------------------------------------------------------
# 2. Normalização per-gas (opcional)
# ---------------------------------------------------------------------------

class ScalerPerGas:
    """MinMaxScaler aplicado independentemente a cada gás.

    Mantém a interface simples (fit / transform / inverse_transform) e
    garante que o fit seja feito exclusivamente nos dados de treino.
    """

    def __init__(self) -> None:
        self.min_ : Optional[np.ndarray] = None  # shape (n_gases,)
        self.scale_: Optional[np.ndarray] = None  # shape (n_gases,)

    def fit(self, series_treino: np.ndarray) -> "ScalerPerGas":
        """Calcula min e max por gás a partir das séries de treino.

        Parameters
        ----------
        series_treino : np.ndarray
            Shape (n_series, T, n_gases).
        """
        # Achatamos (n_series, T) por gás para obter min/max globais de treino
        flat = series_treino.reshape(-1, series_treino.shape[-1])  # (n_series*T, n_gases)
        self.min_   = flat.min(axis=0)   # (n_gases,)
        self.scale_ = flat.max(axis=0) - flat.min(axis=0)  # (n_gases,)
        # Evitar divisão por zero (gás constante)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, series: np.ndarray) -> np.ndarray:
        """Normaliza séries para [0, 1] por gás.

        Parameters
        ----------
        series : np.ndarray
            Shape (n_series, T, n_gases).
        """
        if self.min_ is None:
            raise RuntimeError("Chame fit() antes de transform().")
        return (series - self.min_) / self.scale_

    def inverse_transform(self, series: np.ndarray) -> np.ndarray:
        """Desfaz a normalização (para interpretar erros em escala original).

        Parameters
        ----------
        series : np.ndarray
            Shape (..., n_gases).
        """
        if self.min_ is None:
            raise RuntimeError("Chame fit() antes de inverse_transform().")
        return series * self.scale_ + self.min_


# ---------------------------------------------------------------------------
# 3. Preparação dos tensores (janelamento + normalização)
# ---------------------------------------------------------------------------

def preparar_datasets(
    w: int,
    h: int = LSTM_HORIZONTE,
    normalizar: bool = False,
) -> tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           Optional[ScalerPerGas]]:
    """Prepara tensores (X, y) para treinamento e avaliação da LSTM.

    Parameters
    ----------
    w : int
        Tamanho da janela de entrada (timesteps).
    h : int
        Horizonte de previsão. Default = LSTM_HORIZONTE (1).
    normalizar : bool
        Se True, aplica ScalerPerGas fitado no treino. Default = False.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Tensores prontos para model.fit().
    scaler : ScalerPerGas | None
        Scaler fitado, ou None se normalizar=False.

    Shapes (exemplo com w=20, h=1)
    ------
    X: (n_amostras, 20, 4)   — janela de entrada
    y: (n_amostras, 4)       — próximo timestep (h=1 → squeeze do eixo h)
    """
    s_treino, _, s_val, _, s_teste, _ = carregar_splits()

    scaler: Optional[ScalerPerGas] = None
    if normalizar:
        scaler = ScalerPerGas().fit(s_treino)
        s_treino = scaler.transform(s_treino)
        s_val    = scaler.transform(s_val)
        s_teste  = scaler.transform(s_teste)
        logger.info("Normalização per-gas aplicada (ScalerPerGas).")

    logger.info("Aplicando janelamento w=%d, h=%d...", w, h)
    X_train, y_train = criar_dataset_janelado(s_treino, w=w, h=h)
    X_val,   y_val   = criar_dataset_janelado(s_val,    w=w, h=h)
    X_test,  y_test  = criar_dataset_janelado(s_teste,  w=w, h=h)

    logger.info(
        "Shapes — X_train: %s | X_val: %s | X_test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
