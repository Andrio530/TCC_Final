"""
Fase 5.6 — Verificações obrigatórias antes do feature freeze.

Sustenta quatro afirmações que, hoje, são apenas declarativas no texto:

    1. BASELINE BRUTO (Track B)
       Pergunta: "É a engenharia de features que diferencia os modelos, ou 5 gases
       em ppm já chegam perto de 0,80?"
       Resultado esperado: F1 ≈ 0,78–0,80 (teto informacional dos 5 gases).

    2. TABELA RÓTULO × NORMA
       Pergunta: "Por que o Duval/IEC erra tanto nas classes térmicas?"
       Resultado: % de amostras de cada rótulo na zona Duval/IEC correspondente,
       em treino+val+teste. Confirma §1.3 do parecer (rótulos T3 com química de T1/T2).

    3. TESTES UNITÁRIOS DUVAL E IEC
       Pontos canônicos (vértices/centros de zona publicados) que a implementação
       precisa acertar para eliminar a dúvida "sua implementação está certa?"

    4. REPEATED STRATIFIED K-FOLD 5x5 (Track B)
       Pergunta: "A diferença val=0,704 × teste=0,797 é real ou partição?"
       Resultado: F1 estável com IC estreito, comparável entre C3, C4 e T3,
       usando hiperparâmetros FIXOS (sem re-ajuste por fold).

Uso:
    python -m src.evaluation.fase56
    python -m src.evaluation.fase56 --n-boot 5000 --n-splits 5 --n-repeats 5

Saída:
    reports/metrics/fase56.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import METRICS_DIR, RANDOM_SEED, SEEDS_AVALIACAO  # noqa: E402
from src.evaluation.stats_utils import bootstrap_modelos  # noqa: E402
from src.iga_xgboost.train_classification import (  # noqa: E402
    HPARAMS_DEFAULT,
    LABEL_MAP_B,
    _carregar_csv,
    _treinar_xgboost,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

N_CLASSES_B = len(LABEL_MAP_B)
GASES_BRUTOS_B = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]

# Features do IGA (C3) — carregado do JSON da Fase 5
JSON_FASE5_B = METRICS_DIR / "iga_track_b_results.json"
# Nomes de chave esperados no JSON da Fase 5 Track B
KEY_C3  = "C3_IGA"
KEY_C4  = "C4_default"
KEY_T3  = "T3_Wu2021"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _xy_b(df, feats):
    X = df[feats].values.astype(np.float32)
    y = np.array([LABEL_MAP_B[v] for v in df["fault_type"]], dtype=np.int32)
    return X, y


def _preds_multiseed(df_tr, df_te, feats, hparams):
    Xtr, ytr = _xy_b(df_tr, feats)
    Xte, _ = _xy_b(df_te, feats)
    return np.array([
        _treinar_xgboost(Xtr, ytr, hparams, seed).predict(Xte)
        for seed in SEEDS_AVALIACAO
    ])


# ---------------------------------------------------------------------------
# 1. Baseline bruto (5 gases ppm, sem engenharia de features)
# ---------------------------------------------------------------------------

def baseline_bruto(n_boot: int) -> dict:
    """XGBoost default com apenas os 5 gases brutos (ppm, sem normalização)."""
    logger.info("=== 1. Baseline bruto (5 gases ppm) ===")

    df_tr = _carregar_csv("classification_train.csv")
    df_te = _carregar_csv("classification_test.csv")
    _, y_te = _xy_b(df_te, GASES_BRUTOS_B)

    preds_te = _preds_multiseed(df_tr, df_te, GASES_BRUTOS_B, HPARAMS_DEFAULT)
    f1s = [float(f1_score(y_te, p, average="macro", zero_division=0)) for p in preds_te]

    boot = bootstrap_modelos(y_te, {"bruto_5gases": preds_te}, N_CLASSES_B,
                             n_boot=n_boot, seed=RANDOM_SEED)
    m = boot["modelos"]["bruto_5gases"]
    logger.info("  bruto_5gases  F1=%.3f  IC95%%=[%.3f, %.3f]", m["f1_ponto"], m["ic_inf"], m["ic_sup"])
    return {
        "f1_seeds": f1s,
        "f1_media": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "ic95": [m["ic_inf"], m["ic_sup"]],
        "features": GASES_BRUTOS_B,
        "interpretacao": (
            "F1 próximo de 0,80 confirma o teto informacional: "
            "a engenharia de features não quebra o teto dos 5 gases."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Tabela rótulo × norma (concordância por classe)
# ---------------------------------------------------------------------------

def tabela_rotulo_norma() -> dict:
    """
    Para cada classe, calcula a % de amostras cujo rótulo é a mesma zona
    que Duval ou IEC atribuiria. Usa treino+val+teste em ppm brutos.
    """
    logger.info("=== 2. Tabela rótulo × norma ===")

    from src.iga_xgboost.traditional_methods import predizer_metodos_tradicionais

    dfs = []
    for split in ("train", "val", "test"):
        try:
            dfs.append(_carregar_csv(f"classification_{split}.csv"))
        except Exception:
            pass
    df_all = pd.concat(dfs, ignore_index=True)

    preds = predizer_metodos_tradicionais(df_all)
    y_true = df_all["fault_type"].values

    resultado = {}
    for nome_metodo, y_pred in preds.items():
        por_classe = {}
        for cls in sorted(set(y_true)):
            mask = y_true == cls
            n = mask.sum()
            # % de acerto (rótulo == zona)
            acerto = (y_pred[mask] == cls).sum()
            # % de não-classificados (ND ou DT)
            nd = ((y_pred[mask] == "ND") | (y_pred[mask] == "DT")).sum()
            # distribuição de saídas
            dist = pd.Series(y_pred[mask]).value_counts().to_dict()
            por_classe[cls] = {
                "n": int(n),
                "concordancia_pct": round(100 * acerto / n, 1),
                "nao_classificado_pct": round(100 * nd / n, 1),
                "distribuicao_predita": {str(k): int(v) for k, v in dist.items()},
            }
        resultado[nome_metodo] = por_classe
        logger.info("  %s:", nome_metodo)
        for cls, v in por_classe.items():
            logger.info("    %-8s n=%3d  concordância=%5.1f%%  ND/DT=%5.1f%%  → %s",
                        cls, v["n"], v["concordancia_pct"], v["nao_classificado_pct"],
                        sorted(v["distribuicao_predita"].items(), key=lambda x: -x[1])[:3])
    return resultado


# ---------------------------------------------------------------------------
# 3. Testes unitários Duval e IEC (pontos canônicos)
# ---------------------------------------------------------------------------

def testes_unitarios() -> dict:
    """
    Pontos canônicos extraídos da IEC 60599:2022 Anexo B (Duval Triangle 1)
    e Tabela 1 (razões). Cada ponto deve retornar a zona esperada.
    Falhas são listadas como erros (não geram exceção).
    """
    logger.info("=== 3. Testes unitários Duval e IEC ===")

    from src.iga_xgboost.traditional_methods import classificar_duval, classificar_iec_razoes

    # Duval Triangle 1 – pontos canônicos (CH4, C2H4, C2H2) % da soma dos três
    # Fonte: IEC 60599:2022, Figura B.1 e literatura de Duval
    # Formato: (CH4_ppm, C2H4_ppm, C2H2_ppm, zona_esperada, descricao)
    duval_casos = [
        # PD: %CH4 ≥ 98% → perto do vértice CH4
        (98.0, 1.0, 1.0, "PD", "PD: centro"),
        (99.0, 0.5, 0.5, "PD", "PD: quase puro CH4"),
        # T1: %C2H4 < 20 e %C2H2 < 4
        (80.0, 15.0, 3.0, "T1", "T1: baixo C2H4, C2H2 quase zero"),
        (70.0, 19.0, 2.0, "T1", "T1: limite superior C2H4"),
        # T2: 20 ≤ %C2H4 < 50, %C2H2 < 4
        (60.0, 35.0, 2.0, "T2", "T2: centro"),
        (50.0, 48.0, 2.0, "T2", "T2: limite superior C2H4"),
        # T3: %C2H4 ≥ 50, %C2H2 < 15
        (30.0, 60.0, 5.0, "T3", "T3: centro"),
        (20.0, 72.0, 3.0, "T3", "T3: alto C2H4"),
        # D1: %C2H2 > 13 e %C2H4 < 23
        (20.0, 10.0, 25.0, "D1", "D1: alto C2H2, baixo C2H4"),
        (50.0, 15.0, 40.0, "D1", "D1: centro"),
        # D2: %C2H4 ≥ 23 e %C2H2 > 13
        (20.0, 40.0, 25.0, "D2", "D2: alto C2H4 e C2H2"),
        (10.0, 50.0, 35.0, "D2", "D2: centro"),
        # DT: zona mista — não classificável de forma unívoca; esperado "DT" ou não-PD/T/D
        (30.0, 25.0, 40.0, "DT", "DT: zona mista alta C2H2 e C2H4"),
    ]

    # IEC 60599 Tabela 1 (razões): (H2, CH4, C2H2, C2H4, C2H6, esperado)
    # Formato: gases em ppm (apenas razões importam)
    iec_casos = [
        # PD: C2H2/C2H4 < 0.1, CH4/H2 < 0.1, C2H4/C2H6 < 0.2
        (100, 5, 0.1, 1.0, 10.0, "PD", "PD: all ratios match"),
        # D1: C2H2/C2H4 > 1, CH4/H2 0.1–0.5, C2H4/C2H6 > 1
        (50, 15, 30.0, 20.0, 10.0, "D1", "D1: ratios match"),
        # D2: C2H2/C2H4 0.6–2.5, CH4/H2 0.1–1, C2H4/C2H6 > 2
        (50, 30, 60.0, 100.0, 30.0, "D2", "D2: ratios match"),
        # T1: C2H2/C2H4 < 0.1, CH4/H2 > 1, C2H4/C2H6 < 1
        (10, 50, 0.5, 10.0, 20.0, "T1", "T1: ratios match"),
        # T2: C2H2/C2H4 < 0.1, CH4/H2 > 1, C2H4/C2H6 1–4
        (10, 60, 0.5, 30.0, 10.0, "T2", "T2: ratios match"),
        # T3: C2H2/C2H4 < 0.2, CH4/H2 > 1, C2H4/C2H6 > 4
        (10, 60, 1.0, 60.0, 10.0, "T3", "T3: ratios match"),
        # ND: nenhuma tabela se aplica
        (50, 40, 30.0, 15.0, 5.0, "ND", "ND: nenhum caso da tabela"),
    ]

    resultados_duval = []
    erros_duval = 0
    for ch4, c2h4, c2h2, esperado, desc in duval_casos:
        obtido = classificar_duval(ch4, c2h4, c2h2)
        ok = (obtido == esperado)
        if not ok:
            erros_duval += 1
        resultados_duval.append({
            "descricao": desc, "esperado": esperado, "obtido": obtido, "ok": ok,
            "pct_CH4": round(100*ch4/(ch4+c2h4+c2h2), 1),
            "pct_C2H4": round(100*c2h4/(ch4+c2h4+c2h2), 1),
            "pct_C2H2": round(100*c2h2/(ch4+c2h4+c2h2), 1),
        })

    resultados_iec = []
    erros_iec = 0
    for h2, ch4, c2h2, c2h4, c2h6, esperado, desc in iec_casos:
        obtido = classificar_iec_razoes(h2, ch4, c2h2, c2h4, c2h6)
        ok = (obtido == esperado)
        if not ok:
            erros_iec += 1
        resultados_iec.append({
            "descricao": desc, "esperado": esperado, "obtido": obtido, "ok": ok,
            "r1_C2H2_C2H4": round(c2h2/c2h4, 3) if c2h4 else None,
            "r2_CH4_H2": round(ch4/h2, 3) if h2 else None,
            "r3_C2H4_C2H6": round(c2h4/c2h6, 3) if c2h6 else None,
        })

    logger.info("  Duval: %d/%d corretos", len(duval_casos)-erros_duval, len(duval_casos))
    for r in resultados_duval:
        status = "✓" if r["ok"] else "✗ ERRO"
        logger.info("    %s  %s → esperado=%s obtido=%s (%s%% CH4, %s%% C2H4, %s%% C2H2)",
                    status, r["descricao"], r["esperado"], r["obtido"],
                    r["pct_CH4"], r["pct_C2H4"], r["pct_C2H2"])
    logger.info("  IEC:   %d/%d corretos", len(iec_casos)-erros_iec, len(iec_casos))
    for r in resultados_iec:
        status = "✓" if r["ok"] else "✗ ERRO"
        logger.info("    %s  %s → esperado=%s obtido=%s (r1=%.2f r2=%.2f r3=%.2f)",
                    status, r["descricao"], r["esperado"], r["obtido"],
                    r["r1_C2H2_C2H4"] or 0, r["r2_CH4_H2"] or 0, r["r3_C2H4_C2H6"] or 0)

    return {
        "duval": {"casos": resultados_duval, "erros": erros_duval, "total": len(duval_casos)},
        "iec":   {"casos": resultados_iec,   "erros": erros_iec,   "total": len(iec_casos)},
        "status": "PASS" if (erros_duval == 0 and erros_iec == 0) else "FAIL",
    }


# ---------------------------------------------------------------------------
# 4. Repeated Stratified K-Fold 5×5 (Track B)
# ---------------------------------------------------------------------------

def cv_5x5(n_splits: int = 5, n_repeats: int = 5) -> dict:
    """
    CV repetido no conjunto treino+val (hiperparâmetros FIXOS = default).
    Calcula F1-macro por fold para C3_IGA, C4_default e T3_Wu2021.
    NÃO re-ajusta os hiperparâmetros por fold; os hiperparâmetros de C3
    são carregados do JSON da Fase 5 (fixos).
    """
    logger.info("=== 4. Repeated Stratified K-Fold 5x5 ===")

    with open(JSON_FASE5_B, encoding="utf-8") as f:
        fase5 = json.load(f)

    feats_c3  = fase5["C3_IGA"]["features_selecionadas"]
    hpar_c3   = fase5["C3_IGA"]["hiperparametros"]
    feats_t3  = fase5["T3_Wu2021"]["features_selecionadas"]

    # treino + val (não tocar no teste)
    df_tr = _carregar_csv("china_features_train.csv")
    df_va = _carregar_csv("china_features_val.csv")
    df_tv = pd.concat([df_tr, df_va], ignore_index=True)

    configs = {
        "C3_IGA":     (feats_c3, hpar_c3),
        "C4_default": (fase5["C4_default"]["features_selecionadas"], HPARAMS_DEFAULT),
        "T3_Wu2021":  (feats_t3, HPARAMS_DEFAULT),
        "bruto_5gases": ([c for c in GASES_BRUTOS_B if c in df_tv.columns], HPARAMS_DEFAULT),
    }

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=RANDOM_SEED)
    y_all = np.array([LABEL_MAP_B[v] for v in df_tv["fault_type"]])
    resultado = {}

    for nome, (feats, hp) in configs.items():
        avail = [f for f in feats if f in df_tv.columns]
        if len(avail) < len(feats):
            missing = set(feats) - set(avail)
            logger.warning("  %s: features ausentes em treino+val: %s", nome, missing)
        if not avail:
            logger.warning("  %s: sem features disponíveis, pulando.", nome)
            continue
        X_all = df_tv[avail].values.astype(np.float32)
        f1_folds = []
        for train_idx, val_idx in rskf.split(X_all, y_all):
            Xtr, ytr = X_all[train_idx], y_all[train_idx]
            Xva, yva = X_all[val_idx], y_all[val_idx]
            clf = _treinar_xgboost(Xtr, ytr, hp, RANDOM_SEED)
            p = clf.predict(Xva)
            f1_folds.append(float(f1_score(yva, p, average="macro", zero_division=0)))
        f1_arr = np.array(f1_folds)
        resultado[nome] = {
            "f1_media": float(f1_arr.mean()),
            "f1_std":   float(f1_arr.std()),
            "ic95_normal": [
                float(f1_arr.mean() - 1.96 * f1_arr.std() / np.sqrt(len(f1_arr))),
                float(f1_arr.mean() + 1.96 * f1_arr.std() / np.sqrt(len(f1_arr))),
            ],
            "n_folds": len(f1_folds),
        }
        logger.info("  %s: F1 CV = %.3f ± %.3f  IC95%% [%.3f, %.3f]",
                    nome, f1_arr.mean(), f1_arr.std(),
                    resultado[nome]["ic95_normal"][0], resultado[nome]["ic95_normal"][1])

    return resultado


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(n_boot: int, n_splits: int, n_repeats: int) -> dict:
    resultado = {
        "n_boot":    n_boot,
        "n_splits":  n_splits,
        "n_repeats": n_repeats,
        # Katser dataset metadata (confirmed from labels_rul_train.csv and paper)
        "katser_dataset_nota": (
            "O dataset Katser (2022) possui 1 rótulo por série (FDD = serie-level), "
            "inclui RUL (Remaining Useful Life) e séries de 420 pontos a 12h/ponto. "
            "Os valores de gás estão em escala muito pequena (ordem 1e-3), "
            "consistentes com dados simulados ou de um sistema de monitoramento online "
            "de usina nuclear, não de medições laboratoriais mensais. "
            "Isso implica que o argumento físico sobre a lentidão de falhas térmicas "
            "não pode ser assumido sem verificação no artigo original."
        ),
        "baseline_bruto": baseline_bruto(n_boot),
        "tabela_rotulo_norma": tabela_rotulo_norma(),
        "testes_unitarios": testes_unitarios(),
        "cv_5x5": cv_5x5(n_splits, n_repeats),
    }

    out = METRICS_DIR / "fase56.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_py(resultado), f, ensure_ascii=False, indent=2)
    logger.info("Salvo em %s", out)
    return resultado


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fase 5.6 — verificações obrigatórias")
    ap.add_argument("--n-boot",    type=int, default=2000)
    ap.add_argument("--n-splits",  type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=5)
    args = ap.parse_args()
    main(args.n_boot, args.n_splits, args.n_repeats)
