"""
Fase 5.5 — Track A (Katser, 4 classes): ABLAÇÃO DO LSTM + bootstrap.

Pergunta que este script responde (hipótese central do TCC):
    "As previsões da LSTM (features lstm_*) melhoram o diagnóstico do XGBoost?"

Hoje a Fase 5 não testa isso: o IGA ter escolhido features lstm_* não prova
que elas agregam informação (são muito correlacionadas com os valores
observados obs_*). A ablação compara, no MESMO teste e com as MESMAS seeds:

  Com XGBoost default (sem ajuste de hiperparâmetros):
    C2   : 17 features (obs + razões + LSTM)           [Fase 5]
    C5a  :  9 features (obs + razões)  — SEM LSTM
    C6   :  8 features (somente lstm_*)

  Com IGA (seleção + hiperparâmetros):
    C1   : IGA sobre 17 features                        [Fase 5]
    C5b  : IGA sobre  9 features (sem LSTM)             [fase55_iga_sem_lstm.py]
    C1-L : features de C1 SEM as lstm_* (hparams de C1, sem re-ajuste;
           ablação post-hoc "sem custo", indicada como tal)

Também reporta (a) IC95% por bootstrap estratificado e comparações pareadas;
(b) o fitness de CV (somente treino) de C1 × C5b, que não toca o teste;
(c) diagnóstico de redundância entre lstm_* e obs_* (correlação / R² linear).

Pré-requisito para C5b: rodar antes
    python -m src.evaluation.fase55_iga_sem_lstm

Uso:
    python -m src.evaluation.fase55_track_a
    python -m src.evaluation.fase55_track_a --n-boot 5000

Saída: reports/metrics/fase55_track_a.json
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import METRICS_DIR, RANDOM_SEED, SEEDS_AVALIACAO  # noqa: E402
from src.evaluation.stats_utils import bootstrap_modelos, comparar_pareado  # noqa: E402
from src.iga_xgboost.cromossomo import (  # noqa: E402
    FEATURES_TRACK_A,
    FEATURES_TRACK_A_SEM_LSTM,
)
from src.iga_xgboost.train_hybrid import (  # noqa: E402
    HPARAMS_DEFAULT,
    LABEL_MAP_A,
    _carregar_csv,
    _treinar_xgboost,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASSES = list(LABEL_MAP_A.keys())          # Normal, PD, D1, T1
N_CLASSES = len(CLASSES)
JSON_FASE5 = METRICS_DIR / "iga_track_a_results.json"
JSON_IGA_SEM_LSTM = METRICS_DIR / "fase55_track_a_iga_sem_lstm.json"

FEATURES_SO_LSTM = [f for f in FEATURES_TRACK_A if f.startswith("lstm_")]
GASES = ["H2", "CO", "C2H4", "C2H2"]


def _py(obj):
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _py(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _xy(df, feats):
    X = df[feats].values.astype(np.float32)
    y = np.array([LABEL_MAP_A[v] for v in df["fault_type"]], dtype=np.int32)
    return X, y


def _preds_multiseed(df_tr, df_alvo, feats, hparams, sample_weight) -> np.ndarray:
    Xtr, ytr = _xy(df_tr, feats)
    Xal, _ = _xy(df_alvo, feats)
    return np.array([
        _treinar_xgboost(Xtr, ytr, hparams, seed, sample_weight=sample_weight).predict(Xal)
        for seed in SEEDS_AVALIACAO
    ])


def _f1s(y, preds):
    return [float(f1_score(y, p, average="macro", zero_division=0)) for p in preds]


def _diagnostico_redundancia(df_train) -> dict:
    """Quão previsível é lstm_* a partir de obs_* (mesmo gás)?

    r e R² de uma regressão linear simples lstm ~ obs, no treino. Valores
    próximos de 1 indicam que a feature LSTM é, na prática, uma cópia
    suavizada do valor observado (informação quase redundante).
    """
    saida = {}
    for gas in GASES:
        obs = df_train[f"obs_{gas}"].values.astype(float)
        for h in (14, 28):
            lst = df_train[f"lstm_h{h}_{gas}"].values.astype(float)
            r = float(np.corrcoef(obs, lst)[0, 1])
            saida[f"{gas}_h{h}"] = {"pearson_r": r, "r2_linear": r * r}
    return saida


def _confusao_normal_t1(y, pred) -> dict:
    n, t1 = LABEL_MAP_A["Normal"], LABEL_MAP_A["T1"]
    return {
        "Normal_predito_T1": int(np.sum((y == n) & (pred == t1))),
        "T1_predito_Normal": int(np.sum((y == t1) & (pred == n))),
    }


def main(n_boot: int) -> dict:
    with open(JSON_FASE5, encoding="utf-8") as f:
        fase5 = json.load(f)
    df_tr = _carregar_csv("katser_features_train.csv")
    df_va = _carregar_csv("katser_features_val.csv")
    df_te = _carregar_csv("katser_features_test.csv")
    _, y_tr = _xy(df_tr, FEATURES_TRACK_A)
    _, y_va = _xy(df_va, FEATURES_TRACK_A)
    _, y_te = _xy(df_te, FEATURES_TRACK_A)

    counts = Counter(y_tr)
    sw = np.array([len(y_tr) / (len(counts) * counts[v]) for v in y_tr])

    c1 = fase5["C1_IGA"]
    feats_c1 = c1["features_selecionadas"]
    feats_c1_sem_lstm = [f for f in feats_c1 if not f.startswith("lstm_")]

    # nome → (features, hparams, referência no JSON da Fase 5 para auditoria)
    configs: dict[str, tuple[list[str], dict, list[float] | None]] = {
        "C2_default_17": (FEATURES_TRACK_A, HPARAMS_DEFAULT,
                          fase5["C2_default"]["multiseed"]["f1_por_seed"]),
        "C5a_default_sem_LSTM": (FEATURES_TRACK_A_SEM_LSTM, HPARAMS_DEFAULT, None),
        "C6_default_so_LSTM": (FEATURES_SO_LSTM, HPARAMS_DEFAULT, None),
        "C1_IGA_17": (feats_c1, c1["hiperparametros"], c1["multiseed"]["f1_por_seed"]),
        "C1_menos_LSTM_posthoc": (feats_c1_sem_lstm, c1["hiperparametros"], None),
    }
    iga_sem = None
    if JSON_IGA_SEM_LSTM.exists():
        with open(JSON_IGA_SEM_LSTM, encoding="utf-8") as f:
            iga_sem = json.load(f)
        configs["C5b_IGA_sem_LSTM"] = (
            iga_sem["features_selecionadas"], iga_sem["hiperparametros"], None,
        )
    else:
        logger.warning("%s não encontrado — C5b será omitido. Rode antes: "
                       "python -m src.evaluation.fase55_iga_sem_lstm", JSON_IGA_SEM_LSTM)

    preds_te, preds_va, repro, feats_usadas = {}, {}, {}, {}
    for nome, (feats, hp, ref) in configs.items():
        logger.info("Treinando %s (%d features, 5 seeds)...", nome, len(feats))
        preds_te[nome] = _preds_multiseed(df_tr, df_te, feats, hp, sw)
        preds_va[nome] = _preds_multiseed(df_tr, df_va, feats, hp, sw)
        feats_usadas[nome] = feats
        if ref is not None:
            rec = _f1s(y_te, preds_te[nome])
            repro[nome] = {
                "f1_recomputado": rec, "f1_json_fase5": ref,
                "max_abs_diff": float(np.max(np.abs(np.array(rec) - np.array(ref)))),
            }
            if repro[nome]["max_abs_diff"] > 1e-6:
                logger.warning("%s NÃO reproduz o JSON da Fase 5 (diff=%.2e). "
                               "Provável causa: versões de XGBoost/sklearn diferentes "
                               "das de requirements.txt.", nome, repro[nome]["max_abs_diff"])

    boot = bootstrap_modelos(y_te, preds_te, N_CLASSES, n_boot=n_boot, seed=RANDOM_SEED)

    pares = [
        ("C2_default_17", "C5a_default_sem_LSTM"),    # LSTM agrega, com default?
        ("C2_default_17", "C6_default_so_LSTM"),
        ("C1_IGA_17", "C1_menos_LSTM_posthoc"),       # remover LSTM de C1 (sem re-ajuste)
        ("C1_IGA_17", "C2_default_17"),               # IGA × default (mesmas features cand.)
    ]
    if "C5b_IGA_sem_LSTM" in preds_te:
        pares += [
            ("C1_IGA_17", "C5b_IGA_sem_LSTM"),        # LSTM agrega, com IGA?
            ("C5b_IGA_sem_LSTM", "C5a_default_sem_LSTM"),  # IGA × default sem LSTM
        ]
    comparacoes = [comparar_pareado(boot, a, b) for a, b in pares]

    # F1 por seed e validação
    resumo_cfg = {}
    for nome in configs:
        f1_seeds = _f1s(y_te, preds_te[nome])
        resumo_cfg[nome] = {
            "n_features": len(feats_usadas[nome]),
            "features": feats_usadas[nome],
            "f1_teste_media_seeds": float(np.mean(f1_seeds)),
            "f1_teste_std_seeds": float(np.std(f1_seeds)),
            "f1_validacao_media_seeds": float(np.mean(_f1s(y_va, preds_va[nome]))),
            "ic95_bootstrap_teste": [boot["modelos"][nome]["ic_inf"],
                                     boot["modelos"][nome]["ic_sup"]],
            "confusao_Normal_T1_seed42": _confusao_normal_t1(y_te, preds_te[nome][0]),
        }

    cv_fitness = {"C1_IGA_17": c1.get("fitness_cv")}
    if iga_sem is not None:
        cv_fitness["C5b_IGA_sem_LSTM"] = iga_sem["fitness_cv"]

    resultado = {
        "n_boot": n_boot,
        "n_teste": int(len(y_te)),
        "suporte_teste": {c: int((y_te == i).sum()) for c, i in LABEL_MAP_A.items()},
        "reprodutibilidade_vs_fase5": repro,
        "configs": resumo_cfg,
        "bootstrap": boot["modelos"],
        "comparacoes_pareadas": comparacoes,
        "fitness_cv_somente_treino": cv_fitness,
        "redundancia_lstm_vs_obs_treino": _diagnostico_redundancia(df_tr),
        "iga_sem_lstm": ({k: v for k, v in iga_sem.items()
                          if k not in ("historico_melhor", "historico_media")}
                         if iga_sem else None),
        "nota_C1_menos_LSTM": ("Ablação post-hoc: usa as features escolhidas pelo IGA "
                               "em C1 sem as lstm_*, mantendo os hiperparâmetros de C1 "
                               "(que foram ajustados COM as lstm_*). É um limite "
                               "inferior para 'IGA sem LSTM'; C5b é o teste justo."),
    }
    out = METRICS_DIR / "fase55_track_a.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_py(resultado), f, ensure_ascii=False, indent=2)
    logger.info("Salvo em %s", out)

    logger.info("-" * 60)
    for nome, m in boot["modelos"].items():
        logger.info("  %-24s F1=%.3f  IC95%%=[%.3f, %.3f]",
                    nome, m["f1_ponto"], m["ic_inf"], m["ic_sup"])
    for c in comparacoes:
        logger.info("  Δ %s − %s = %+.3f  IC95%%=[%+.3f, %+.3f]  exclui0=%s",
                    c["a"], c["b"], c["delta_f1_ponto"], c["ic_inf"], c["ic_sup"],
                    c["ic_exclui_zero"])
    return resultado


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fase 5.5 — Track A (ablação LSTM)")
    ap.add_argument("--n-boot", type=int, default=2000)
    main(ap.parse_args().n_boot)
