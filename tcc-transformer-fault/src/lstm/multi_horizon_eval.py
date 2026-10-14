"""
Avaliacao de multiplos horizontes de previsao recursiva -- LSTM (Fase 3).

Mede a degradacao da qualidade de previsao conforme o horizonte aumenta,
usando previsao recursiva (iterada): a saida do passo t e usada como parte
da entrada do passo t+1.

Horizontes testados (alinhados com o artigo de referencia):
    h=1   ->  12h     (baseline)
    h=2   ->  24h     (1 dia)
    h=4   ->  48h     (2 dias)
    h=7   ->  84h     (3.5 dias)
    h=14  ->  168h    (1 semana  - Semana 1 do artigo)
    h=28  ->  336h    (2 semanas - Semana 2 do artigo)
    h=42  ->  504h    (3 semanas - Semana 3 do artigo)
    h=56  ->  672h    (4 semanas - Semana 4 do artigo)
    h=70  ->  840h    (5 semanas - Semana 5 do artigo, limite >85%)
    h=84  ->  1008h   (6 semanas - Semana 6 do artigo, inicio queda)
    h=98  ->  1176h   (7 semanas - Semana 7 do artigo)
    h=112 ->  1344h   (8 semanas)
    h=126 ->  1512h   (9 semanas)
    h=140 ->  1680h   (10 semanas - estimativa do ponto de corte R2~0.80)
    h=154 ->  1848h   (11 semanas)
    h=168 ->  2016h   (12 semanas - limite superior de busca)

Criterio de corte (horizonte maximo aceitavel):
    R2 medio entre gases >= R2_THRESHOLD (padrao: 0.80)

Uso:
    python -m src.lstm.multi_horizon_eval --normalizar
    python -m src.lstm.multi_horizon_eval
    python -m src.lstm.multi_horizon_eval --normalizar --seeds 42 123 456 789 1234

Saidas (todas salvas separadamente):
    reports/metrics/lstm_multi_horizon_results{sufixo}.json
    reports/figures/lstm_multi_horizon_r2{sufixo}.png
    reports/figures/lstm_multi_horizon_mae_pct{sufixo}.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DATA_PROCESSED,
    FIGURES_DIR,
    KATSER_GASES,
    METRICS_DIR,
    SEEDS_AVALIACAO,
)
from src.lstm.data_loader import ScalerPerGas, carregar_splits  # noqa: E402
from src.lstm.model import fixar_seeds  # noqa: E402
from src.lstm.trainer import treinar_modelo  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

HORIZONTES: list[int] = [1, 2, 4, 7, 14, 28, 42, 56, 70, 84, 98, 112, 126]
HORAS_POR_PASSO: int = 12
SEMANAS_POR_PASSO: float = HORAS_POR_PASSO / (7 * 24)
R2_THRESHOLD: float = 0.80
MAE_PCT_THRESHOLD: float = 20.0
N_SERIES_TESTE: int = 200


# ---------------------------------------------------------------------------
# 1. Previsao recursiva
# ---------------------------------------------------------------------------

def previsao_recursiva(model, janela_inicial: np.ndarray, h: int) -> np.ndarray:
    """Gera h previsoes a frente usando previsao recursiva.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo LSTM treinado.
    janela_inicial : np.ndarray
        Shape (w, n_gases).
    h : int
        Numero de passos a prever.

    Returns
    -------
    np.ndarray, shape (h, n_gases)
    """
    w, n_gases = janela_inicial.shape
    janela = janela_inicial.copy().astype(np.float32)
    previsoes = np.zeros((h, n_gases), dtype=np.float32)
    for passo in range(h):
        x_input = janela[np.newaxis, :, :]
        y_hat = model.predict(x_input, verbose=0)
        previsoes[passo] = y_hat[0]
        janela = np.roll(janela, -1, axis=0)
        janela[-1] = y_hat[0]
    return previsoes


# ---------------------------------------------------------------------------
# 2. Calculo de metricas por gas
# ---------------------------------------------------------------------------

def _calcular_mae_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, dict]:
    """Calcula MAE e R2 por gas."""
    erros = y_true - y_pred
    mae: dict[str, float] = {}
    r2: dict[str, float] = {}
    for i, g in enumerate(KATSER_GASES):
        mae[g] = float(np.mean(np.abs(erros[:, i])))
        ss_res = float(np.sum(erros[:, i] ** 2))
        ss_tot = float(np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2))
        r2[g] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, r2


# ---------------------------------------------------------------------------
# 3. Avaliacao de um horizonte com uma seed
# ---------------------------------------------------------------------------

def avaliar_horizonte(
    model,
    series_teste: np.ndarray,
    w: int,
    h: int,
    scaler: Optional[ScalerPerGas],
    n_series: int = N_SERIES_TESTE,
    seed: int = 42,
) -> dict:
    """Avalia o modelo no horizonte h usando previsao recursiva.

    Para cada serie de teste:
      - Usa os primeiros w timesteps como janela de entrada.
      - Gera h previsoes recursivas.
      - Compara a previsao do passo h com o valor real na posicao w+h-1.

    Salva metricas em escala normalizada E em escala original (ppm),
    separadamente, para nao perder informacao.
    """
    rng = np.random.default_rng(seed)
    n_disponivel = series_teste.shape[0]
    n_amostrar = min(n_series, n_disponivel)
    indices = rng.choice(n_disponivel, size=n_amostrar, replace=False)

    T = series_teste.shape[1]
    max_inicio = T - w - h
    if max_inicio <= 0:
        logger.warning("h=%d: series muito curtas (T=%d, w=%d). Pulando.", h, T, w)
        nan_dict = {g: float("nan") for g in KATSER_GASES}
        return {
            "mae_por_gas_norm": nan_dict.copy(),
            "r2_por_gas_norm":  nan_dict.copy(),
            "mae_por_gas_orig": nan_dict.copy(),
            "r2_por_gas_orig":  nan_dict.copy(),
            "mae_medio_orig":   float("nan"),
            "r2_medio_orig":    float("nan"),
            "mae_pct_por_gas":  nan_dict.copy(),
        }

    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []

    for idx in indices:
        serie = series_teste[idx]
        inicio = rng.integers(0, max_inicio + 1)
        janela_ini = serie[inicio: inicio + w]
        verdadeiro  = serie[inicio + w + h - 1]
        previsoes = previsao_recursiva(model, janela_ini, h)
        y_pred_h = previsoes[-1]
        y_true_list.append(verdadeiro)
        y_pred_list.append(y_pred_h)

    y_true_norm = np.array(y_true_list)
    y_pred_norm = np.array(y_pred_list)

    # Metricas em escala normalizada (salvas separadamente)
    mae_norm, r2_norm = _calcular_mae_r2(y_true_norm, y_pred_norm)

    # Desnormaliza para escala original (ppm)
    if scaler is not None:
        y_true_orig = scaler.inverse_transform(y_true_norm)
        y_pred_orig = scaler.inverse_transform(y_pred_norm)
    else:
        y_true_orig = y_true_norm
        y_pred_orig = y_pred_norm

    # Metricas em escala original (ppm)
    mae_orig, r2_orig = _calcular_mae_r2(y_true_orig, y_pred_orig)

    # MAE% por gas: erro relativo a media do gas no conjunto de teste
    mae_pct: dict[str, float] = {}
    for i, g in enumerate(KATSER_GASES):
        media_gas = float(np.mean(np.abs(y_true_orig[:, i])))
        mae_pct[g] = float(mae_orig[g] / media_gas * 100) if media_gas > 0 else float("nan")

    r2_medio = float(np.nanmean([r2_orig[g] for g in KATSER_GASES]))
    mae_medio = float(np.mean([mae_orig[g] for g in KATSER_GASES]))

    return {
        "mae_por_gas_norm": mae_norm,
        "r2_por_gas_norm":  r2_norm,
        "mae_por_gas_orig": mae_orig,
        "r2_por_gas_orig":  r2_orig,
        "mae_medio_orig":   mae_medio,
        "r2_medio_orig":    r2_medio,
        "mae_pct_por_gas":  mae_pct,
    }


# ---------------------------------------------------------------------------
# 4. Avaliacao completa com multiplas seeds
# ---------------------------------------------------------------------------

def avaliar_multiplos_horizontes(
    w: int,
    json_path: Path,
    normalizar: bool = True,
    seeds: list[int] = SEEDS_AVALIACAO,
    horizontes: list[int] = HORIZONTES,
    n_series: int = N_SERIES_TESTE,
) -> dict:
    """Avalia o modelo em todos os horizontes com multiplas seeds.

    Para cada horizonte h, retreina o modelo com cada seed e agrega
    os resultados (media +/- desvio padrao entre seeds).
    Salva progresso incremental em json_path.
    """
    _, _, _, _, s_teste, _ = carregar_splits()

    resultados = None
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                resultados = json.load(f)
            if resultados.get("w_star") != w or resultados.get("normalizar") != normalizar:
                logger.warning("Configuracoes mudaram. Ignorando progresso salvo e recomecando.")
                resultados = None
            else:
                logger.info("Progresso anterior carregado de %s", json_path.name)
        except Exception as e:
            logger.warning("Erro ao ler %s: %s. Recomecando.", json_path.name, e)
            resultados = None

    if resultados is None:
        resultados = {
            "w_star": w,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "n_series_por_seed": n_series,
            "horizontes_passos": horizontes,
            "horas_por_passo": HORAS_POR_PASSO,
            "r2_threshold": R2_THRESHOLD,
            "mae_pct_threshold": MAE_PCT_THRESHOLD,
            "normalizar": normalizar,
            "por_horizonte": {},
        }

    for h in horizontes:
        h_horas = h * HORAS_POR_PASSO
        h_semanas = h * SEMANAS_POR_PASSO
        
        if str(h) in resultados["por_horizonte"]:
            logger.info("=== Horizonte h=%d (%dh = %.1f semanas) [JA CONCLUIDO] ===", h, h_horas, h_semanas)
            continue
            
        logger.info("=== Horizonte h=%d (%dh = %.1f semanas) ===", h, h_horas, h_semanas)

        r2_medio_seeds:    list[float] = []
        mae_medio_seeds:   list[float] = []
        r2_por_gas_seeds:  dict[str, list[float]] = {g: [] for g in KATSER_GASES}
        mae_por_gas_seeds: dict[str, list[float]] = {g: [] for g in KATSER_GASES}
        mae_pct_seeds:     dict[str, list[float]] = {g: [] for g in KATSER_GASES}

        for seed in seeds:
            logger.info("  seed=%d | h=%d", seed, h)
            fixar_seeds(seed)
            model, _, scaler = treinar_modelo(w=w, seed=seed, normalizar=normalizar, verbose=0)

            if scaler is not None:
                s_teste_scaled = scaler.transform(s_teste)
            else:
                s_teste_scaled = s_teste
                scaler = None

            res = avaliar_horizonte(
                model=model, series_teste=s_teste_scaled, w=w, h=h,
                scaler=scaler, n_series=n_series, seed=seed,
            )

            r2_medio_seeds.append(res["r2_medio_orig"])
            mae_medio_seeds.append(res["mae_medio_orig"])
            for g in KATSER_GASES:
                r2_por_gas_seeds[g].append(res["r2_por_gas_orig"][g])
                mae_por_gas_seeds[g].append(res["mae_por_gas_orig"][g])
                mae_pct_seeds[g].append(res["mae_pct_por_gas"][g])

        resultados["por_horizonte"][str(h)] = {
            "h_passos":  h,
            "h_horas":   h_horas,
            "h_semanas": round(h_semanas, 2),
            "r2_medio": {
                "media": float(np.nanmean(r2_medio_seeds)),
                "std":   float(np.nanstd(r2_medio_seeds)),
            },
            "mae_medio_orig": {
                "media": float(np.nanmean(mae_medio_seeds)),
                "std":   float(np.nanstd(mae_medio_seeds)),
            },
            "r2_por_gas": {
                g: {"media": float(np.nanmean(r2_por_gas_seeds[g])),
                    "std":   float(np.nanstd(r2_por_gas_seeds[g]))}
                for g in KATSER_GASES
            },
            "mae_por_gas_orig": {
                g: {"media": float(np.nanmean(mae_por_gas_seeds[g])),
                    "std":   float(np.nanstd(mae_por_gas_seeds[g]))}
                for g in KATSER_GASES
            },
            "mae_pct_por_gas": {
                g: {"media": float(np.nanmean(mae_pct_seeds[g])),
                    "std":   float(np.nanstd(mae_pct_seeds[g]))}
                for g in KATSER_GASES
            },
        }

        r2_h = resultados["por_horizonte"][str(h)]["r2_medio"]["media"]
        logger.info("  h=%d | R2 medio=%.4f | %s", h, r2_h,
                    "ACEITAVEL" if r2_h >= R2_THRESHOLD else "ABAIXO DO THRESHOLD")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
        logger.info("  Progresso salvo para h=%d", h)

    horizonte_max = _determinar_horizonte_maximo(resultados)
    resultados["horizonte_max_aceitavel"] = horizonte_max
    logger.info(
        ">>> Horizonte maximo aceitavel (R2>=%.2f): h=%d (%dh = %.1f semanas)",
        R2_THRESHOLD, horizonte_max["h_passos"], horizonte_max["h_horas"], horizonte_max["h_semanas"],
    )
    return resultados


def _determinar_horizonte_maximo(resultados: dict) -> dict:
    """Identifica o maior horizonte onde R2 medio >= R2_THRESHOLD."""
    por_h = resultados["por_horizonte"]
    h_validos = [int(h) for h, d in por_h.items() if d["r2_medio"]["media"] >= R2_THRESHOLD]
    h_max = max(h_validos) if h_validos else 1
    h_horas = h_max * HORAS_POR_PASSO
    h_semanas = h_max * SEMANAS_POR_PASSO
    r2_no_max = por_h[str(h_max)]["r2_medio"]["media"]
    return {
        "h_passos":  h_max,
        "h_horas":   h_horas,
        "h_semanas": round(h_semanas, 2),
        "r2_medio":  round(r2_no_max, 4),
        "criterio":  f"R2 medio >= {R2_THRESHOLD}",
    }


# ---------------------------------------------------------------------------
# 5. Graficos
# ---------------------------------------------------------------------------

def _eixo_x_duplo(ax, hs: list[int]) -> None:
    """Adiciona eixo X superior com equivalencia em semanas."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    horas = [h * HORAS_POR_PASSO for h in hs]
    semanas = [h * SEMANAS_POR_PASSO for h in hs]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(horas)
    ax2.set_xticklabels([f"{s:.1f}sem" for s in semanas], fontsize=8, rotation=45)
    ax2.set_xlabel("Equivalencia em semanas (comparacao com artigo de referencia)")
    return ax2


def plotar_r2_degradacao(resultados: dict, sufixo: str = "") -> Path:
    """Grafico de degradacao do R2 por gas e R2 medio vs horizonte.

    Inclui:
      - Uma linha por gas (R2 medio entre seeds com banda +/- 1 sigma)
      - Linha do R2 medio entre gases
      - Linha de corte em R2_THRESHOLD
      - Marcador do horizonte maximo aceitavel
      - Eixo X duplo: horas e semanas
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    destino = FIGURES_DIR / f"lstm_multi_horizon_r2{sufixo}.png"

    por_h = resultados["por_horizonte"]
    hs = [int(k) for k in por_h.keys()]
    horas = [h * HORAS_POR_PASSO for h in hs]

    cores = {"H2": "#2196F3", "CH4": "#FF5722", "CO": "#4CAF50", "C2H2": "#9C27B0"}
    cores_fb = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(14, 7))

    # R2 por gas
    for i, g in enumerate(KATSER_GASES):
        cor = cores.get(g, cores_fb[i % 4])
        r2_m = [por_h[str(h)]["r2_por_gas"][g]["media"] for h in hs]
        r2_s = [por_h[str(h)]["r2_por_gas"][g]["std"]   for h in hs]
        ax.plot(horas, r2_m, "o-", color=cor, label=f"R2 {g}", linewidth=2, markersize=5)
        ax.fill_between(horas, [m - s for m, s in zip(r2_m, r2_s)],
                        [m + s for m, s in zip(r2_m, r2_s)], alpha=0.12, color=cor)

    # R2 medio
    r2_med_m = [por_h[str(h)]["r2_medio"]["media"] for h in hs]
    r2_med_s = [por_h[str(h)]["r2_medio"]["std"]   for h in hs]
    ax.plot(horas, r2_med_m, "s--", color="black", label="R2 medio (todos gases)",
            linewidth=2.5, markersize=7, zorder=5)
    ax.fill_between(horas, [m - s for m, s in zip(r2_med_m, r2_med_s)],
                    [m + s for m, s in zip(r2_med_m, r2_med_s)], alpha=0.08, color="black")

    # Linha de corte
    ax.axhline(R2_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold R2={R2_THRESHOLD}", zorder=4)

    # Marcador horizonte maximo
    h_max = resultados["horizonte_max_aceitavel"]
    hm_p = h_max["h_passos"]
    hm_h = h_max["h_horas"]
    hm_s = h_max["h_semanas"]
    ax.axvline(hm_h, color="darkgreen", linestyle=":", linewidth=2, zorder=4,
               label=f"Horizonte max aceitavel: h={hm_p}, {hm_h}h ({hm_s:.1f} semanas)")

    _eixo_x_duplo(ax, hs)
    ax.set_xlabel("Horizonte de previsao (horas)")
    ax.set_ylabel("R2 (coeficiente de determinacao)")
    n_s = resultados["n_seeds"]
    n_sp = resultados["n_series_por_seed"]
    ax.set_title(f"Degradacao do R2 por Horizonte -- LSTM (previsao recursiva)\nn_seeds={n_s} | n_series/seed={n_sp}")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grafico R2 salvo em %s", destino)
    return destino


def plotar_mae_pct_degradacao(resultados: dict, sufixo: str = "") -> Path:
    """Grafico de degradacao do MAE-pct por gas vs horizonte.

    Inclui:
      - Uma linha por gas (MAE% medio +/- 1 sigma)
      - Linha de corte em MAE_PCT_THRESHOLD
      - Eixo X duplo: horas e semanas
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    destino = FIGURES_DIR / f"lstm_multi_horizon_mae_pct{sufixo}.png"

    por_h = resultados["por_horizonte"]
    hs = [int(k) for k in por_h.keys()]
    horas = [h * HORAS_POR_PASSO for h in hs]

    cores = {"H2": "#2196F3", "CH4": "#FF5722", "CO": "#4CAF50", "C2H2": "#9C27B0"}
    cores_fb = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, g in enumerate(KATSER_GASES):
        cor = cores.get(g, cores_fb[i % 4])
        mae_m = [por_h[str(h)]["mae_pct_por_gas"][g]["media"] for h in hs]
        mae_s = [por_h[str(h)]["mae_pct_por_gas"][g]["std"]   for h in hs]
        ax.plot(horas, mae_m, "o-", color=cor, label=f"MAE-pct {g}", linewidth=2, markersize=5)
        ax.fill_between(horas, [max(0, m - s) for m, s in zip(mae_m, mae_s)],
                        [m + s for m, s in zip(mae_m, mae_s)], alpha=0.12, color=cor)

    ax.axhline(MAE_PCT_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold MAE-pct={MAE_PCT_THRESHOLD}%", zorder=4)

    _eixo_x_duplo(ax, hs)
    ax.set_xlabel("Horizonte de previsao (horas)")
    ax.set_ylabel("MAE-pct (erro relativo a media do gas, %)")
    n_s = resultados["n_seeds"]
    n_sp = resultados["n_series_por_seed"]
    ax.set_title(f"Degradacao do MAE-pct por Horizonte -- LSTM (previsao recursiva)\nn_seeds={n_s} | n_series/seed={n_sp}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grafico MAE-pct salvo em %s", destino)
    return destino


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avaliacao de multiplos horizontes -- LSTM (previsao recursiva)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--normalizar", action="store_true", default=False)
    parser.add_argument("--w", type=int, default=None, metavar="JANELA")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, metavar="SEED")
    parser.add_argument("--n-series", type=int, default=N_SERIES_TESTE, metavar="N")
    parser.add_argument("--suffix", type=str, default=None, metavar="SUFIXO")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    sufixo = args.suffix if args.suffix is not None else ("_norm" if args.normalizar else "")

    if args.w is not None:
        w = args.w
    else:
        cfg_name = "lstm_best_config_norm.json" if args.normalizar else "lstm_best_config.json"
        cfg_path = DATA_PROCESSED / cfg_name
        if not cfg_path.exists():
            cfg_path = DATA_PROCESSED / "lstm_best_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{cfg_path} nao encontrado. "
                "Execute run_lstm.py --train-final primeiro, ou passe --w explicitamente."
            )
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        w = int(cfg["w_star"])

    seeds = args.seeds if args.seeds is not None else SEEDS_AVALIACAO

    logger.info("=== AVALIACAO DE MULTIPLOS HORIZONTES ===")
    logger.info("w=%d | normalizar=%s | seeds=%s | sufixo=%r", w, args.normalizar, seeds, sufixo)
    logger.info("Horizontes: %s", HORIZONTES)
    logger.info("Threshold R2: %.2f | Threshold MAE-pct: %.1f%%", R2_THRESHOLD, MAE_PCT_THRESHOLD)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = METRICS_DIR / f"lstm_multi_horizon_results{sufixo}.json"

    resultados = avaliar_multiplos_horizontes(
        w=w, json_path=json_path, normalizar=args.normalizar, seeds=seeds,
        horizontes=HORIZONTES, n_series=args.n_series,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    logger.info("Resultados salvos em %s", json_path)

    plotar_r2_degradacao(resultados, sufixo=sufixo)
    plotar_mae_pct_degradacao(resultados, sufixo=sufixo)

    h_max = resultados["horizonte_max_aceitavel"]
    logger.info("=" * 60)
    logger.info("RESULTADO FINAL")
    logger.info("  Horizonte maximo aceitavel: h=%d passos", h_max["h_passos"])
    logger.info("  Equivalente a: %dh (%.1f semanas)", h_max["h_horas"], h_max["h_semanas"])
    logger.info("  R2 medio no ponto de corte: %.4f", h_max["r2_medio"])
    logger.info("  Criterio: %s", h_max["criterio"])
    logger.info("=" * 60)
    
    # Força o download no Colab ao finalizar
    try:
        from google.colab import files
        logger.info("Baixando JSON e gráficos automaticamente no Colab...")
        files.download(str(json_path))
        fig_r2 = FIGURES_DIR / f"lstm_multi_horizon_r2{sufixo}.png"
        fig_mae = FIGURES_DIR / f"lstm_multi_horizon_mae_pct{sufixo}.png"
        if fig_r2.exists():
            files.download(str(fig_r2))
        if fig_mae.exists():
            files.download(str(fig_mae))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
