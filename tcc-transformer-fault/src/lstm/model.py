"""
Arquitetura da LSTM para previsão de gases dissolvidos (Estágio 1 — Fase 3).

Especificação (Seção 3.5 do TCC):
  - 2 camadas LSTM empilhadas (LSTM_UNITS = 64 cada)
  - Dropout entre camadas
  - Saída Dense para os 4 gases (H2, CO, C2H4, C2H2)
  - Otimizador: Adam com learning rate fixo
  - Loss: MSE (minimizado durante treino; MAE e RMSE avaliados separadamente)
"""

import os
import random

import numpy as np

# TensorFlow importado de forma lazy para não aumentar tempo de import em testes
# que não precisam do modelo.


def fixar_seeds(seed: int) -> None:
    """Fixa todas as seeds para garantir reprodutibilidade total.

    Deve ser chamada antes de construir o modelo e antes de treinar.

    Parameters
    ----------
    seed : int
        Valor da seed (use RANDOM_SEED da config para a run principal).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf  # noqa: PLC0415

    tf.random.set_seed(seed)


def construir_modelo(
    w: int,
    n_features: int = 4,
) :
    """Constrói e compila o modelo LSTM empilhado de 2 camadas.

    Parameters
    ----------
    w : int
        Tamanho da janela de entrada (número de timesteps).
    n_features : int
        Número de gases (features) por timestep. Default = 4 (base Katser).

    Returns
    -------
    model : tf.keras.Model
        Modelo compilado, pronto para .fit().

    Architecture
    ------------
    Input(w, n_features)
      → LSTM(LSTM_UNITS, return_sequences=True)
      → Dropout(LSTM_DROPOUT)
      → LSTM(LSTM_UNITS, return_sequences=False)
      → Dropout(LSTM_DROPOUT)
      → Dense(n_features)        ← previsão de 1 passo à frente (h=1)
    """
    import sys
    from pathlib import Path

    import tensorflow as tf  # noqa: PLC0415

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.config import (  # noqa: PLC0415
        LSTM_DROPOUT,
        LSTM_LEARNING_RATE,
        LSTM_UNITS,
    )

    inputs = tf.keras.Input(shape=(w, n_features), name="gas_window")

    # Camada LSTM 1: retorna sequência completa para empilhar
    x = tf.keras.layers.LSTM(
        LSTM_UNITS,
        return_sequences=True,
        name="lstm_1",
    )(inputs)
    x = tf.keras.layers.Dropout(LSTM_DROPOUT, name="dropout_1")(x)

    # Camada LSTM 2: retorna apenas o último hidden state
    x = tf.keras.layers.LSTM(
        LSTM_UNITS,
        return_sequences=False,
        name="lstm_2",
    )(x)
    x = tf.keras.layers.Dropout(LSTM_DROPOUT, name="dropout_2")(x)

    # Saída: previsão dos n_features gases no próximo timestep
    outputs = tf.keras.layers.Dense(n_features, name="gas_forecast")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_dga_forecast")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    return model
