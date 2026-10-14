"""
Treinamento da LSTM: busca de janela ótima e avaliação com múltiplas seeds.

Fluxo principal:
  1. buscar_melhor_janela() — treina para cada w ∈ JANELAS_CANDIDATAS com
     seed=42, avalia no VALIDAÇÃO e seleciona o w* com menor MAE agregado.
  2. avaliar_com_multiplas_seeds() — treina o modelo final w* vezes com
     seeds diferentes e avalia no TESTE OFICIAL (900 séries).
  3. treinar_modelo() — função base usada pelos dois acima.

Regra de ouro (validation_checklist.md):
  - O conjunto de TESTE nunca é usado para selecionar w nem hiperparâmetros.
  - O SCALER (se usado) é sempre fitado apenas nos dados de TREINO.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DATA_PROCESSED,
    JANELAS_CANDIDATAS,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS_MAX,
    LSTM_HORIZONTE,
    LSTM_MIN_DELTA,
    LSTM_PATIENCE,
    METRICS_DIR,
    RANDOM_SEED,
    SEEDS_AVALIACAO,
)
from src.lstm.data_loader import ScalerPerGas, preparar_datasets  # noqa: E402
from src.lstm.evaluator import calcular_metricas  # noqa: E402
from src.lstm.model import construir_modelo, fixar_seeds  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Treinar um único modelo
# ---------------------------------------------------------------------------

def treinar_modelo(
    w: int,
    seed: int = RANDOM_SEED,
    normalizar: bool = False,
    verbose: int = 0,
):
    """Treina a LSTM com os hiperparâmetros do config.py.

    Parameters
    ----------
    w : int
        Tamanho da janela de entrada.
    seed : int
        Seed para reprodutibilidade (fixa TF, NumPy, Python).
    normalizar : bool
        Se True, aplica ScalerPerGas nos dados.
    verbose : int
        Verbosidade do Keras (0 = silencioso, 1 = progresso, 2 = por época).

    Returns
    -------
    model : tf.keras.Model
        Modelo treinado com os melhores pesos (restore_best_weights=True).
    history : tf.keras.callbacks.History
        Histórico de métricas por época.
    scaler : ScalerPerGas | None
        Scaler fitado no treino, ou None se normalizar=False.
    """
    import tensorflow as tf  # noqa: PLC0415

    fixar_seeds(seed)

    X_train, y_train, X_val, y_val, _, _, scaler = preparar_datasets(
        w=w, h=LSTM_HORIZONTE, normalizar=normalizar
    )

    model = construir_modelo(w=w, n_features=X_train.shape[-1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=LSTM_PATIENCE,
            min_delta=LSTM_MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS_MAX,
        batch_size=LSTM_BATCH_SIZE,
        callbacks=callbacks,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))
    logger.info(
        "w=%d seed=%d | best_epoch=%d | val_loss=%.6f | tempo=%.1fs",
        w, seed, best_epoch, best_val_loss, elapsed,
    )

    return model, history, scaler


# ---------------------------------------------------------------------------
# 2. Busca do melhor tamanho de janela (avaliado na VALIDAÇÃO)
# ---------------------------------------------------------------------------

def buscar_melhor_janela(
    janelas: list[int] = JANELAS_CANDIDATAS,
    normalizar: bool = False,
    verbose: int = 0,
) -> tuple[int, dict]:
    """Treina para cada w em `janelas` e seleciona o melhor por MAE no VAL.

    O conjunto de TESTE não é tocado nesta etapa.

    Parameters
    ----------
    janelas : list[int]
        Tamanhos de janela a testar (default: JANELAS_CANDIDATAS = [10,20,30,50]).
    normalizar : bool
        Repassado para treinar_modelo().
    verbose : int
        Verbosidade do Keras.

    Returns
    -------
    w_star : int
        Janela com menor MAE agregado no validação.
    resultados : dict
        Métricas completas por janela (salvas em reports/metrics/).
    """
    from src.config import KATSER_GASES  # noqa: PLC0415

    resultados: dict = {}

    for w in janelas:
        logger.info("=== Buscando janela w=%d (seed=%d) ===", w, RANDOM_SEED)

        model, _, scaler = treinar_modelo(
            w=w, seed=RANDOM_SEED, normalizar=normalizar, verbose=verbose
        )

        # Avalia no VALIDAÇÃO (nunca no teste)
        _, _, X_val, y_val, _, _, _ = preparar_datasets(
            w=w, h=LSTM_HORIZONTE, normalizar=normalizar
        )
        y_pred_val = model.predict(X_val, verbose=0)

        metricas_val = calcular_metricas(y_val, y_pred_val, nomes_gases=KATSER_GASES)
        resultados[str(w)] = {
            "mae_agregado": metricas_val["mae_agregado"],
            "rmse_agregado": metricas_val["rmse_agregado"],
            "mae_por_gas": metricas_val["mae_por_gas"],
            "rmse_por_gas": metricas_val["rmse_por_gas"],
        }

        logger.info(
            "w=%d | MAE_val=%.6f | RMSE_val=%.6f",
            w, metricas_val["mae_agregado"], metricas_val["rmse_agregado"],
        )

    # Salva resultados de busca
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    saida = METRICS_DIR / "lstm_window_search.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    logger.info("Resultados da busca salvos em %s", saida)

    # Seleciona melhor w por MAE agregado na validação
    w_star = int(min(resultados, key=lambda k: resultados[k]["mae_agregado"]))
    logger.info("Melhor janela: w*=%d (MAE_val=%.6f)", w_star, resultados[str(w_star)]["mae_agregado"])

    return w_star, resultados


# ---------------------------------------------------------------------------
# 3. Avaliação final com múltiplas seeds no TESTE
# ---------------------------------------------------------------------------

def avaliar_com_multiplas_seeds(
    w: int,
    seeds: list[int] = SEEDS_AVALIACAO,
    normalizar: bool = False,
    verbose: int = 0,
) -> dict:
    """Treina o modelo final `len(seeds)` vezes e avalia no TESTE OFICIAL.

    Reporta média ± desvio padrão de MAE e RMSE por gás e agregado,
    conforme exigido pelo item 2 do validation_checklist.md.

    Parameters
    ----------
    w : int
        Janela selecionada (w*) pela busca.
    seeds : list[int]
        Lista de seeds para avaliação (default: SEEDS_AVALIACAO).
    normalizar : bool
        Repassado para treinar_modelo().
    verbose : int
        Verbosidade do Keras.

    Returns
    -------
    resultado_final : dict
        Métricas agregadas por seed + média ± σ. Salvo em reports/metrics/.
    """
    import tensorflow as tf  # noqa: PLC0415
    from src.config import KATSER_GASES  # noqa: PLC0415

    runs: list[dict] = []
    melhor_val_loss = float("inf")
    melhor_model = None
    melhor_scaler: Optional[ScalerPerGas] = None

    for seed in seeds:
        logger.info("=== Seed %d (w=%d) ===", seed, w)

        model, history, scaler = treinar_modelo(
            w=w, seed=seed, normalizar=normalizar, verbose=verbose
        )

        # Avalia no TESTE (apenas aqui — nunca antes)
        _, _, _, _, X_test, y_test, _ = preparar_datasets(
            w=w, h=LSTM_HORIZONTE, normalizar=normalizar
        )
        y_pred_test = model.predict(X_test, verbose=0)

        # Desnormaliza antes de calcular métricas → MAE/RMSE em escala original (ppm)
        if scaler is not None:
            y_true_orig = scaler.inverse_transform(y_test)
            y_pred_orig = scaler.inverse_transform(y_pred_test)
        else:
            y_true_orig = y_test
            y_pred_orig = y_pred_test

        metricas_teste = calcular_metricas(y_true_orig, y_pred_orig, nomes_gases=KATSER_GASES)
        metricas_teste["seed"] = seed
        runs.append(metricas_teste)

        logger.info(
            "seed=%d | MAE_teste=%.6f | RMSE_teste=%.6f",
            seed, metricas_teste["mae_agregado"], metricas_teste["rmse_agregado"],
        )

        # Guarda melhor modelo (por val_loss) e respectivo scaler
        val_loss_final = float(np.min(history.history["val_loss"]))
        if val_loss_final < melhor_val_loss:
            melhor_val_loss = val_loss_final
            melhor_model = model
            melhor_scaler = scaler

    # Agrega estatísticas
    maes_agr  = [r["mae_agregado"]  for r in runs]
    rmses_agr = [r["rmse_agregado"] for r in runs]

    mae_por_gas_runs  = {g: [r["mae_por_gas"][g]  for r in runs] for g in KATSER_GASES}
    rmse_por_gas_runs = {g: [r["rmse_por_gas"][g] for r in runs] for g in KATSER_GASES}

    resultado_final = {
        "w_star": w,
        "seeds": seeds,
        "n_runs": len(seeds),
        "mae_agregado_media": float(np.mean(maes_agr)),
        "mae_agregado_std":   float(np.std(maes_agr)),
        "rmse_agregado_media": float(np.mean(rmses_agr)),
        "rmse_agregado_std":   float(np.std(rmses_agr)),
        "mae_por_gas": {
            g: {"media": float(np.mean(mae_por_gas_runs[g])),
                "std":   float(np.std(mae_por_gas_runs[g]))}
            for g in KATSER_GASES
        },
        "rmse_por_gas": {
            g: {"media": float(np.mean(rmse_por_gas_runs[g])),
                "std":   float(np.std(rmse_por_gas_runs[g]))}
            for g in KATSER_GASES
        },
        "runs": runs,
    }

    # Salva resultados
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    saida = METRICS_DIR / "lstm_test_results.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado_final, f, ensure_ascii=False, indent=2)
    logger.info("Resultados finais salvos em %s", saida)

    # Persiste melhor modelo e config
    if melhor_model is not None:
        modelo_path = DATA_PROCESSED / "lstm_best_model.keras"
        melhor_model.save(modelo_path)
        logger.info("Melhor modelo salvo em %s", modelo_path)

    config_path = DATA_PROCESSED / "lstm_best_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "w_star": w,
            "normalizar": normalizar,
            "seeds_avaliacao": seeds,
            "gases": KATSER_GASES,
            "horizonte": LSTM_HORIZONTE,
        }, f, ensure_ascii=False, indent=2)

    return resultado_final
