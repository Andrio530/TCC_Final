"""
Fase 5.5 — Track B (base chinesa, 7 classes): avaliação honesta.

O que este script faz (NENHUM modelo é re-otimizado; features e hiperparâmetros
vêm dos JSONs da Fase 5, escolhidos só com o treino):

  1. Reconstrói C3 (IGA), C4 (default) e T3 (Wu 2021) com as MESMAS features,
     hiperparâmetros, SMOTE e 5 seeds da Fase 5, e confere que reproduzem os
     F1 por seed gravados no JSON (auditoria de reprodutibilidade).
  2. Avalia as regras IEC 60599 (razões) e Duval corrigidas (ver
     src/iga_xgboost/traditional_methods.py).
  3. Bootstrap estratificado e pareado no teste: IC 95% do F1-macro de cada
     modelo e IC da DIFERENÇA entre pares de modelos. Duas visões:
       - sete_classes  : todas as amostras de teste;
       - somente_falhas: só amostras cujo rótulo real é falha (6 classes) —
         comparação justa com as regras, que não definem a classe Normal.
  4. Mede o quanto o F1 do MESMO modelo muda entre validação e teste
     (indica a incerteza amostral real, ignorada pelo "± seed").
  5. Lista as principais confusões de C3 (seed 42) no teste.
  6. (opcional --cv N) Validação cruzada estratificada 5×N repetida, para
     comparar C3 × T3 × C4 sem depender de uma única partição.

Uso:
  python -m src.evaluation.fase55_track_b
  python -m src.evaluation.fase55_track_b --cv 3       # + CV repetida 5x3 (lento)
  python -m src.evaluation.fase55_track_b --n-boot 5000

Saídas: reports/metrics/fase55_track_b.json
        reports/metrics/fase55_track_b_repeated_cv.json (com --cv)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, METRICS_DIR, RANDOM_SEED, SEEDS_AVALIACAO  # noqa: E402
from src.evaluation.stats_utils import (  # noqa: E402
    bootstrap_modelos,
    comparar_pareado,
    teste_t_corrigido_nadeau_bengio,
)
from src.iga_xgboost.cromossomo import FEATURES_TRACK_B, FEATURES_TRACK_WU  # noqa: E402
from src.iga_xgboost.traditional_methods import predizer_metodos_tradicionais  # noqa: E402
from src.iga_xgboost.train_classification import (  # noqa: E402
    HPARAMS_DEFAULT,
    LABEL_MAP_B,
    CustomSMOTE,
    _treinar_xgboost,
    carregar_dados as carregar_dados_china,
)
from src.iga_xgboost.train_wu2021_baseline import (  # noqa: E402
    WuCustomSMOTE,
    carregar_e_preparar_dados as carregar_dados_wu,
)

import pandas as pd  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASSES = list(LABEL_MAP_B.keys())            # Normal, PD, D1, D2, T1, T2, T3
CLASSES_FALHA = CLASSES[1:]                   # 6 classes de falha
N_CLASSES = len(CLASSES)
JSON_FASE5 = METRICS_DIR / "iga_track_b_results.json"


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def _py(obj):
    """Converte tipos numpy para tipos nativos (serialização JSON)."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _py(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _preds_multiseed(X_tr, y_tr, X_te, hparams, idx_feat) -> np.ndarray:
    """(n_seeds, n_teste) predições, uma linha por seed de SEEDS_AVALIACAO."""
    return np.array([
        _treinar_xgboost(X_tr[:, idx_feat], y_tr, hparams, seed)
        .predict(X_te[:, idx_feat])
        for seed in SEEDS_AVALIACAO
    ])


def _f1_por_seed(y, preds) -> list[float]:
    return [float(f1_score(y, p, average="macro", zero_division=0)) for p in preds]


def _mapear_regra_para_int(pred_str: np.ndarray) -> np.ndarray:
    """Rótulo textual → inteiro; ND/DT → N_CLASSES ("fora das classes")."""
    return np.array([LABEL_MAP_B.get(p, N_CLASSES) for p in pred_str], dtype=np.int64)


def _visao_somente_falhas(y: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Restringe a amostras de falha e remapeia rótulos 1..6 → 0..5.

    Predição 'Normal' (0) e qualquer coisa fora das classes viram 6 ("fora"),
    ou seja, contam como erro sem formar classe extra.
    """
    mask = y != LABEL_MAP_B["Normal"]
    y_f = y[mask] - 1
    p = np.atleast_2d(preds)[:, mask]
    p_f = np.where((p >= 1) & (p <= 6), p - 1, 6)
    return y_f, p_f


def _top_confusoes(y: np.ndarray, pred: np.ndarray, k: int = 8) -> list[dict]:
    pares = {}
    for t, p in zip(y, pred):
        if t != p:
            pares[(t, p)] = pares.get((t, p), 0) + 1
    inv = {v: n for n, v in LABEL_MAP_B.items()}
    ordenado = sorted(pares.items(), key=lambda kv: -kv[1])[:k]
    return [{"real": inv[t], "predito": inv[p], "n": n} for (t, p), n in ordenado]


# ---------------------------------------------------------------------------
# Avaliação principal (partição única treino/val/teste da Fase 5)
# ---------------------------------------------------------------------------

def avaliar(n_boot: int) -> dict:
    with open(JSON_FASE5, encoding="utf-8") as f:
        fase5 = json.load(f)

    # --- Dados (mesmos arquivos e mesmos SMOTE da Fase 5) ----------------
    Xc_tr, yc_tr, Xc_va, yc_va, Xc_te, yc_te = carregar_dados_china()
    Xw_tr, yw_tr, Xw_va, yw_va, Xw_te, yw_te, wu_pipe = carregar_dados_wu()
    if not (np.array_equal(yc_te, yw_te) and np.array_equal(yc_va, yw_va)
            and np.array_equal(yc_tr, yw_tr)):
        raise RuntimeError(
            "Ordem das amostras difere entre china_features_* e "
            "classification_* — comparação pareada inválida."
        )
    y_te, y_va = yc_te, yc_va

    smote_c = CustomSMOTE(scaler_path=DATA_PROCESSED / "china_feature_scaler.pkl",
                          random_state=RANDOM_SEED, k_neighbors=5)
    Xc_sm, yc_sm = smote_c.fit_resample(Xc_tr, yc_tr)
    smote_w = WuCustomSMOTE(pipeline_path=wu_pipe, random_state=RANDOM_SEED, k_neighbors=5)
    Xw_sm, yw_sm = smote_w.fit_resample(Xw_tr, yw_tr)

    # --- Modelos (features/hparams vêm da Fase 5) ------------------------
    cfg = {
        "C3_IGA": dict(src=fase5["C3_IGA"], feats=FEATURES_TRACK_B, dados="c"),
        "C4_default": dict(src=None, feats=FEATURES_TRACK_B, dados="c"),
        "T3_Wu2021": dict(src=fase5["T3_Wu2021"], feats=FEATURES_TRACK_WU, dados="w"),
    }
    preds_te, preds_va, repro = {}, {}, {}
    for nome, c in cfg.items():
        if c["src"] is None:                                   # default: todas as features
            feats_ativas, hp = c["feats"], HPARAMS_DEFAULT
            ref = fase5["C4_default"]["multiseed"]["f1_por_seed"]
        else:
            feats_ativas = c["src"]["features_selecionadas"]
            hp = c["src"]["hiperparametros"]
            ref = c["src"]["multiseed"]["f1_por_seed"]
        idx = [c["feats"].index(f) for f in feats_ativas]
        Xtr, ytr, Xva, Xte = (
            (Xc_sm, yc_sm, Xc_va, Xc_te) if c["dados"] == "c"
            else (Xw_sm, yw_sm, Xw_va, Xw_te)
        )
        logger.info("Treinando %s (%d features, 5 seeds)...", nome, len(idx))
        preds_te[nome] = _preds_multiseed(Xtr, ytr, Xte, hp, idx)
        preds_va[nome] = _preds_multiseed(Xtr, ytr, Xva, hp, idx)
        f1_seeds = _f1_por_seed(y_te, preds_te[nome])
        repro[nome] = {
            "f1_recomputado": f1_seeds,
            "f1_json_fase5": ref,
            "max_abs_diff": float(np.max(np.abs(np.array(f1_seeds) - np.array(ref)))),
        }
        if repro[nome]["max_abs_diff"] > 1e-6:
            logger.warning("%s NÃO reproduz exatamente o JSON da Fase 5 (diff=%.2e). "
                           "Provável causa: versão do XGBoost/sklearn diferente da "
                           "usada na Fase 5 (ver requirements.txt).",
                           nome, repro[nome]["max_abs_diff"])

    # --- Regras (gases brutos) -------------------------------------------
    df_raw = pd.read_csv(DATA_PROCESSED / "classification_test.csv")
    if not np.array_equal(np.array([LABEL_MAP_B[v] for v in df_raw["fault_type"]]), y_te):
        raise RuntimeError("classification_test.csv fora de ordem em relação ao teste.")
    regras_str = predizer_metodos_tradicionais(df_raw)
    regras = {k: _mapear_regra_para_int(v)[None, :] for k, v in regras_str.items()}

    # --- Bootstrap: sete classes -----------------------------------------
    todos_te = {**preds_te, **regras}
    boot7 = bootstrap_modelos(y_te, todos_te, N_CLASSES, n_boot=n_boot, seed=RANDOM_SEED)

    # --- Bootstrap: somente falhas ---------------------------------------
    y_f = None
    preds_f = {}
    for nome, p in todos_te.items():
        y_f, preds_f[nome] = _visao_somente_falhas(y_te, p)
    boot6 = bootstrap_modelos(y_f, preds_f, len(CLASSES_FALHA), n_boot=n_boot, seed=RANDOM_SEED)

    pares = [
        ("C3_IGA", "T3_Wu2021"), ("C3_IGA", "C4_default"), ("T3_Wu2021", "C4_default"),
        ("C3_IGA", "Duval"), ("C3_IGA", "IEC60599"),
    ]
    comp7 = [comparar_pareado(boot7, a, b) for a, b in pares]
    comp6 = [comparar_pareado(boot6, a, b) for a, b in pares]

    # --- Validação × teste do mesmo modelo --------------------------------
    boot_va = bootstrap_modelos(y_va, {"C3_IGA": preds_va["C3_IGA"]}, N_CLASSES,
                                n_boot=n_boot, seed=RANDOM_SEED)
    f1_va = boot_va["modelos"]["C3_IGA"]
    f1_te = boot7["modelos"]["C3_IGA"]

    # --- Regras: cobertura e métricas (reaproveita o módulo corrigido) ----
    from src.iga_xgboost.traditional_methods import avaliar_metodos_tradicionais  # noqa: PLC0415
    regras_det = avaliar_metodos_tradicionais(DATA_PROCESSED / "classification_test.csv")
    for v in regras_det.values():
        v.pop("predicoes", None)

    resultado = {
        "n_boot": n_boot,
        "n_teste": int(len(y_te)),
        "suporte_teste": {c: int((y_te == i).sum()) for c, i in LABEL_MAP_B.items()},
        "reprodutibilidade_vs_fase5": repro,
        "bootstrap_sete_classes": {
            "modelos": boot7["modelos"], "comparacoes_pareadas": comp7,
        },
        "bootstrap_somente_falhas": {
            "n_amostras": int(len(y_f)),
            "modelos": boot6["modelos"], "comparacoes_pareadas": comp6,
        },
        "validacao_vs_teste_C3": {
            "f1_validacao": f1_va, "f1_teste": f1_te,
            "diferenca_ponto": f1_te["f1_ponto"] - f1_va["f1_ponto"],
            "ics_se_sobrepoem": bool(f1_va["ic_inf"] <= f1_te["ic_sup"]
                                     and f1_te["ic_inf"] <= f1_va["ic_sup"]),
        },
        "regras_tradicionais": regras_det,
        "top_confusoes_C3_seed42_teste": _top_confusoes(y_te, preds_te["C3_IGA"][0]),
        "f1_std_entre_seeds": {
            n: float(np.std(_f1_por_seed(y_te, p))) for n, p in preds_te.items()
        },
    }
    out = METRICS_DIR / "fase55_track_b.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_py(resultado), f, ensure_ascii=False, indent=2)
    logger.info("Salvo em %s", out)
    _log_resumo(resultado)
    return resultado


def _log_resumo(r: dict) -> None:
    logger.info("-" * 60)
    for visao in ("bootstrap_sete_classes", "bootstrap_somente_falhas"):
        logger.info("[%s]", visao)
        for nome, m in r[visao]["modelos"].items():
            logger.info("  %-11s F1=%.3f  IC95%%=[%.3f, %.3f]",
                        nome, m["f1_ponto"], m["ic_inf"], m["ic_sup"])
        for c in r[visao]["comparacoes_pareadas"]:
            logger.info("  Δ %s − %s = %+.3f  IC95%%=[%+.3f, %+.3f]  exclui0=%s",
                        c["a"], c["b"], c["delta_f1_ponto"], c["ic_inf"], c["ic_sup"],
                        c["ic_exclui_zero"])
    v = r["validacao_vs_teste_C3"]
    logger.info("C3 val=%.3f [%.3f,%.3f] | teste=%.3f [%.3f,%.3f] | ICs sobrepõem=%s",
                v["f1_validacao"]["f1_ponto"], v["f1_validacao"]["ic_inf"],
                v["f1_validacao"]["ic_sup"], v["f1_teste"]["f1_ponto"],
                v["f1_teste"]["ic_inf"], v["f1_teste"]["ic_sup"], v["ics_se_sobrepoem"])


# ---------------------------------------------------------------------------
# Validação cruzada repetida (opcional)
# ---------------------------------------------------------------------------

def validacao_cruzada_repetida(n_repeats: int, n_splits: int = 5) -> dict:
    """CV estratificada 5×n_repeats sobre treino+val+teste (rótulos originais).

    ⚠️ Viés declarado: features e hiperparâmetros de C3/T3 foram escolhidos pelo
    IGA usando as linhas de TREINO da partição original. Aqui essas linhas
    reaparecem em folds de teste, então C3 e T3 ficam OTIMISTAS frente a C4
    (default, sem ajuste). A comparação C3 × T3 permanece justa (ambos foram
    ajustados pelo mesmo procedimento, nas mesmas linhas). Use este resultado
    para medir a VARIABILIDADE do F1 entre partições e para C3 × T3; não o use
    para afirmar que o IGA supera o default.

    Os scalers (log1p + MinMax) foram ajustados no treino original e são
    aplicados a todas as linhas: transformação monotônica, sem uso de rótulos;
    efeito apenas no espaço de interpolação do SMOTE.
    """
    with open(JSON_FASE5, encoding="utf-8") as f:
        fase5 = json.load(f)

    Xc_tr, yc_tr, Xc_va, yc_va, Xc_te, yc_te = carregar_dados_china()
    Xw_tr, yw_tr, Xw_va, yw_va, Xw_te, yw_te, wu_pipe = carregar_dados_wu()
    X_c = np.vstack([Xc_tr, Xc_va, Xc_te])
    X_w = np.vstack([Xw_tr, Xw_va, Xw_te])
    y = np.concatenate([yc_tr, yc_va, yc_te])
    y_w = np.concatenate([yw_tr, yw_va, yw_te])
    if not np.array_equal(y, y_w):
        raise RuntimeError("Ordem de amostras inconsistente entre as duas fontes.")

    smote_c = CustomSMOTE(scaler_path=DATA_PROCESSED / "china_feature_scaler.pkl",
                          random_state=RANDOM_SEED, k_neighbors=5)
    smote_w = WuCustomSMOTE(pipeline_path=wu_pipe, random_state=RANDOM_SEED, k_neighbors=5)

    c3 = fase5["C3_IGA"]
    t3 = fase5["T3_Wu2021"]
    idx_c3 = [FEATURES_TRACK_B.index(f) for f in c3["features_selecionadas"]]
    idx_c4 = list(range(len(FEATURES_TRACK_B)))
    idx_t3 = [FEATURES_TRACK_WU.index(f) for f in t3["features_selecionadas"]]

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=RANDOM_SEED)
    f1 = {"C3_IGA": [], "C4_default": [], "T3_Wu2021": []}
    acc = {k: [] for k in f1}
    n_tr_med, n_te_med = [], []
    t0 = time.time()
    for fold, (tr, te) in enumerate(rskf.split(X_c, y), start=1):
        Xc_f, yc_f = smote_c.fit_resample(X_c[tr], y[tr])
        Xw_f, yw_f = smote_w.fit_resample(X_w[tr], y[tr])
        n_tr_med.append(len(tr)); n_te_med.append(len(te))
        modelos = {
            "C3_IGA": (Xc_f, yc_f, X_c[te], idx_c3, c3["hiperparametros"]),
            "C4_default": (Xc_f, yc_f, X_c[te], idx_c4, HPARAMS_DEFAULT),
            "T3_Wu2021": (Xw_f, yw_f, X_w[te], idx_t3, t3["hiperparametros"]),
        }
        for nome, (Xtr, ytr, Xte, idx, hp) in modelos.items():
            p = _treinar_xgboost(Xtr[:, idx], ytr, hp, RANDOM_SEED).predict(Xte[:, idx])
            f1[nome].append(float(f1_score(y[te], p, average="macro", zero_division=0)))
            acc[nome].append(float(np.mean(p == y[te])))
        logger.info("fold %d/%d | C3=%.3f C4=%.3f T3=%.3f | %.0fs", fold,
                    n_splits * n_repeats, f1["C3_IGA"][-1], f1["C4_default"][-1],
                    f1["T3_Wu2021"][-1], time.time() - t0)

    n_tr, n_te = int(np.mean(n_tr_med)), int(np.mean(n_te_med))
    resumo = {
        n: {"f1_media": float(np.mean(v)), "f1_std_entre_folds": float(np.std(v, ddof=1)),
            "f1_min": float(np.min(v)), "f1_max": float(np.max(v)),
            "acc_media": float(np.mean(acc[n])), "f1_por_fold": v}
        for n, v in f1.items()
    }
    comparacoes = []
    for a, b in [("C3_IGA", "T3_Wu2021"), ("C3_IGA", "C4_default"), ("T3_Wu2021", "C4_default")]:
        d = np.array(f1[a]) - np.array(f1[b])
        comparacoes.append({"a": a, "b": b,
                            **teste_t_corrigido_nadeau_bengio(d, n_tr, n_te)})
    resultado = {
        "n_splits": n_splits, "n_repeats": n_repeats,
        "n_treino_medio": n_tr, "n_teste_medio": n_te,
        "aviso_vies": ("C3 e T3 foram ajustados pelo IGA com linhas que reaparecem "
                       "nos folds de teste → otimistas frente a C4. C3×T3 é justa."),
        "modelos": resumo, "comparacoes_t_corrigido": comparacoes,
    }
    out = METRICS_DIR / "fase55_track_b_repeated_cv.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_py(resultado), f, ensure_ascii=False, indent=2)
    logger.info("Salvo em %s", out)
    return resultado


def main() -> None:
    ap = argparse.ArgumentParser(description="Fase 5.5 — Track B")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cv", type=int, default=0,
                    help="Repetições da CV estratificada 5-fold (0 = não executa).")
    args = ap.parse_args()
    avaliar(args.n_boot)
    if args.cv > 0:
        validacao_cruzada_repetida(args.cv)


if __name__ == "__main__":
    main()
