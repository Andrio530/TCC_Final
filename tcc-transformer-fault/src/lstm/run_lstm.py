"""
Orquestrador do pipeline LSTM (Fase 3).

Uso:
    # Busca de janela ótima (seed=42, avalia no VAL, NÃO toca no teste)
    python -m src.lstm.run_lstm --search-window

    # Treinamento final com 5 seeds (avalia no TESTE oficial)
    python -m src.lstm.run_lstm --train-final

    # Treinamento final com normalização per-gas (recomendação P0)
    python -m src.lstm.run_lstm --train-final --normalizar

    # Pipeline completo (busca + treinamento final)
    python -m src.lstm.run_lstm

    # Pipeline completo com normalização
    python -m src.lstm.run_lstm --normalizar

    # Gerar previsões para a Fase 4 (requer lstm_best_model.keras já salvo)
    python -m src.lstm.run_lstm --generate-predictions

Flags:
    --search-window         Apenas busca de w* na validação (seed=42)
    --train-final           Apenas treinamento final com múltiplas seeds no teste
    --generate-predictions  Apenas geração de previsões para a Fase 4
    --normalizar            Aplica ScalerPerGas per-gas (MinMaxScaler por gás, fit no treino)
    --w W                   Força uso de uma janela específica (ignora lstm_best_config.json)
    --suffix SUFIXO         Sufixo nos arquivos de saída (ex: "_norm"). Se --normalizar
                            e sem --suffix, usa "_norm" automaticamente.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Garante que a raiz do projeto está no sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DATA_PROCESSED,
    FIGURES_DIR,
    JANELAS_CANDIDATAS,
    KATSER_GASES,
    LSTM_HORIZONTE,
    METRICS_DIR,
    SEEDS_AVALIACAO,
)
from src.lstm.data_loader import preparar_datasets  # noqa: E402
from src.lstm.evaluator import (  # noqa: E402
    calcular_metricas,
    plotar_comparacao_janelas,
    plotar_curvas_treinamento,
    plotar_distribuicao_erros,
    plotar_previsoes_amostra,
)
from src.lstm.trainer import (  # noqa: E402
    avaliar_com_multiplas_seeds,
    buscar_melhor_janela,
    treinar_modelo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_lstm")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _carregar_w_star() -> int:
    """Lê w* salvo em lstm_best_config.json."""
    cfg_path = DATA_PROCESSED / "lstm_best_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} não encontrado. "
            "Execute --search-window primeiro para selecionar w*."
        )
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg["w_star"])


def _salvar_config(w: int, normalizar: bool, seeds: list[int], sufixo: str) -> None:
    """Persiste configuração do experimento em lstm_best_config{sufixo}.json."""
    cfg_path = DATA_PROCESSED / f"lstm_best_config{sufixo}.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "w_star": w,
                "normalizar": normalizar,
                "seeds_avaliacao": seeds,
                "gases": KATSER_GASES,
                "horizonte": LSTM_HORIZONTE,
                "sufixo": sufixo,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Configuração salva em %s", cfg_path)


# ---------------------------------------------------------------------------
# Passos do pipeline
# ---------------------------------------------------------------------------

def passo_busca_janela(normalizar: bool, sufixo: str) -> int:
    """Passo 1: busca de w* na validação."""
    logger.info("=== PASSO 1: Busca de janela ótima (seed=42, VAL) ===")
    w_star, resultados = buscar_melhor_janela(
        janelas=JANELAS_CANDIDATAS,
        normalizar=normalizar,
    )

    # Salva comparação de janelas com sufixo
    saida_json = METRICS_DIR / f"lstm_window_search{sufixo}.json"
    with open(saida_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    logger.info("Resultados da busca salvos em %s", saida_json)

    # Gráfico comparativo
    fig_path = FIGURES_DIR / f"lstm_window_comparison{sufixo}.png"
    plotar_comparacao_janelas(resultados, destino=fig_path)

    logger.info(">>> w* selecionado: %d", w_star)
    return w_star


def passo_treinamento_final(w: int, normalizar: bool, sufixo: str) -> dict:
    """Passo 2: treinamento final com 5 seeds no TESTE oficial."""
    logger.info("=== PASSO 2: Treinamento final (w=%d, %d seeds, TESTE) ===",
                w, len(SEEDS_AVALIACAO))

    resultado = avaliar_com_multiplas_seeds(
        w=w,
        seeds=SEEDS_AVALIACAO,
        normalizar=normalizar,
    )

    # Renomeia arquivos de saída com sufixo (avaliar_com_multiplas_seeds salva
    # sempre em lstm_test_results.json e lstm_best_model.keras)
    if sufixo:
        _renomear_saidas(sufixo)

    _salvar_config(w, normalizar, SEEDS_AVALIACAO, sufixo)

    # Gráficos — seed=42 (primeira seed) como referência visual
    _gerar_graficos_pos_treinamento(w=w, normalizar=normalizar, sufixo=sufixo)

    logger.info(
        ">>> MAE agregado: %.6f ± %.6f",
        resultado["mae_agregado_media"],
        resultado["mae_agregado_std"],
    )
    return resultado


def _renomear_saidas(sufixo: str) -> None:
    """Move os arquivos padrão para versões com sufixo (evita duplicatas sem sufixo)."""
    import shutil

    mapa = {
        METRICS_DIR / "lstm_test_results.json":
            METRICS_DIR / f"lstm_test_results{sufixo}.json",
        DATA_PROCESSED / "lstm_best_model.keras":
            DATA_PROCESSED / f"lstm_best_model{sufixo}.keras",
    }
    for origem, destino in mapa.items():
        if origem.exists():
            shutil.move(str(origem), str(destino))
            logger.info("Renomeado %s → %s", origem.name, destino.name)


def _gerar_graficos_pos_treinamento(w: int, normalizar: bool, sufixo: str) -> None:
    """Gera curva de loss + previsões + distribuição de erros para seed=42."""
    logger.info("Gerando gráficos visuais (seed=42)...")

    seed_ref = 42
    model, history, _ = treinar_modelo(w=w, seed=seed_ref, normalizar=normalizar)

    # Curva de treinamento
    fig_loss = FIGURES_DIR / f"lstm_loss_w{w}_seed{seed_ref}{sufixo}.png"
    plotar_curvas_treinamento(history, w=w, seed=seed_ref, destino=fig_loss)

    # Previsões e distribuição de erros no teste
    _, _, _, _, X_test, y_test, _ = preparar_datasets(
        w=w, h=LSTM_HORIZONTE, normalizar=normalizar
    )
    y_pred = model.predict(X_test, verbose=0)

    fig_pred = FIGURES_DIR / f"lstm_predictions_sample{sufixo}.png"
    plotar_previsoes_amostra(y_test, y_pred, destino=fig_pred)

    fig_err = FIGURES_DIR / f"lstm_error_distribution{sufixo}.png"
    plotar_distribuicao_erros(y_test, y_pred, destino=fig_err)

    logger.info("Gráficos salvos em %s", FIGURES_DIR)


def passo_gerar_previsoes(normalizar: bool, sufixo: str) -> None:
    """Passo 3: gera previsões 1-passo-à-frente para a Fase 4."""
    import numpy as np
    import tensorflow as tf

    logger.info("=== PASSO 3: Geração de previsões para a Fase 4 ===")

    # Carrega melhor modelo (com ou sem sufixo)
    modelo_norm = DATA_PROCESSED / f"lstm_best_model{sufixo}.keras"
    modelo_padrao = DATA_PROCESSED / "lstm_best_model.keras"
    modelo_path = modelo_norm if modelo_norm.exists() else modelo_padrao

    if not modelo_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {modelo_path}. "
            "Execute --train-final primeiro."
        )

    model = tf.keras.models.load_model(modelo_path)
    logger.info("Modelo carregado de %s", modelo_path)

    w = _carregar_w_star()
    _, _, _, _, X_test, y_test, _ = preparar_datasets(
        w=w, h=LSTM_HORIZONTE, normalizar=normalizar
    )

    y_pred = model.predict(X_test, verbose=0)

    # Salva previsões para uso na Fase 4
    saida = DATA_PROCESSED / f"lstm_predictions_test{sufixo}.npz"
    np.savez_compressed(saida, y_true=y_test, y_pred=y_pred)
    logger.info("Previsões salvas em %s (shape: %s)", saida, y_pred.shape)

    metricas = calcular_metricas(y_test, y_pred, nomes_gases=KATSER_GASES)
    logger.info(
        "Métricas finais — MAE: %.6f | RMSE: %.6f",
        metricas["mae_agregado"],
        metricas["rmse_agregado"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline LSTM — Fase 3 (previsão de gases dissolvidos)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Modos de execução
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument(
        "--search-window",
        action="store_true",
        help="Apenas busca de w* na validação (seed=42, ~30-60 min com GPU)",
    )
    grupo.add_argument(
        "--train-final",
        action="store_true",
        help="Apenas treinamento final com múltiplas seeds no teste oficial",
    )
    grupo.add_argument(
        "--generate-predictions",
        action="store_true",
        help="Apenas gera previsões 1-passo-à-frente para a Fase 4",
    )

    # Opções de configuração
    parser.add_argument(
        "--normalizar",
        action="store_true",
        default=False,
        help=(
            "Aplica ScalerPerGas (MinMaxScaler per-gas, fit no treino). "
            "Recomendado para corrigir o R² negativo do C2H2. "
            "Os arquivos de saída recebem sufixo '_norm' automaticamente."
        ),
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,
        metavar="JANELA",
        help="Força uso de um tamanho de janela específico (padrão: lê de lstm_best_config.json)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        metavar="SUFIXO",
        help=(
            "Sufixo nos arquivos de saída (ex: '_exp2'). "
            "Se --normalizar e sem --suffix, usa '_norm' automaticamente."
        ),
    )

    args = parser.parse_args()

    # Determina sufixo automaticamente se --normalizar sem --suffix explícito
    if args.suffix is not None:
        sufixo = args.suffix
    elif args.normalizar:
        sufixo = "_norm"
    else:
        sufixo = ""

    if sufixo:
        logger.info("Sufixo de saída: '%s'", sufixo)
    if args.normalizar:
        logger.info("Normalização per-gas (ScalerPerGas) ATIVADA")
    else:
        logger.info("Normalização per-gas: desativada (dados em escala original)")

    # ---------------------------------------------------------------------------
    # Execução dos passos
    # ---------------------------------------------------------------------------

    if args.search_window:
        passo_busca_janela(normalizar=args.normalizar, sufixo=sufixo)

    elif args.train_final:
        w = args.w if args.w is not None else _carregar_w_star()
        logger.info("Usando w=%d para treinamento final", w)
        passo_treinamento_final(w=w, normalizar=args.normalizar, sufixo=sufixo)

    elif args.generate_predictions:
        passo_gerar_previsoes(normalizar=args.normalizar, sufixo=sufixo)

    else:
        # Pipeline completo: busca → treinamento → gráficos
        logger.info("=== PIPELINE COMPLETO ===")
        w_star = passo_busca_janela(normalizar=args.normalizar, sufixo=sufixo)
        passo_treinamento_final(w=w_star, normalizar=args.normalizar, sufixo=sufixo)

    logger.info("=== CONCLUÍDO ===")


if __name__ == "__main__":
    main()