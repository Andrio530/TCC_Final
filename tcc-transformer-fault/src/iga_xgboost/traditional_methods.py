"""
Métodos tradicionais de diagnóstico por DGA para comparação com o IGA-XGBoost.

Fase 5.5 — REESCRITO. A versão anterior continha erros de implementação
(fronteiras de zona do Triângulo de Duval incorretas; "Rogers" que misturava
a tabela de Rogers com as classes IEC; casos não enquadrados convertidos em
"Normal"). Esta versão implementa:

  1. Método das razões da IEC 60599:2015 (Tabela 1) — razões
       R1 = C2H2/C2H4,  R2 = CH4/H2,  R3 = C2H4/C2H6
     → PD, D1, D2, T1, T2, T3 ou "ND" (não classificado: combinação de razões
       fora de todas as faixas da tabela — algo que a própria norma admite).
  2. Triângulo 1 de Duval (IEC 60599:2015, Anexo B) com as zonas
     PD, T1, T2, T3, D1, D2 e DT (zona mista, "não conclusiva" para as 6
     classes do dataset).

Nenhum dos dois métodos define a classe Normal: ambos pressupõem que já existe
uma falha (a IEC 60599 só os aplica quando as concentrações excedem os valores
típicos). Por isso a avaliação é reportada de duas formas:
  - "somente_falhas": apenas as amostras cujo rótulo real é falha (6 classes).
    É a comparação justa para esses métodos.
  - "sete_classes": todas as amostras; qualquer amostra Normal é, por
    construção, classificada como falha (recall de Normal = 0). Reportada
    apenas por transparência.
Em ambos os casos, saídas "ND"/"DT" contam como ERRO (não são ocultadas), e a
cobertura (fração das amostras de falha com diagnóstico em uma das 6 classes)
é reportada separadamente.

Estes métodos são aplicados sobre os gases brutos em PPM (pré-normalização).

Referências:
  - IEC 60599:2015, "Mineral oil-filled electrical equipment in service –
    Guidance on the interpretation of dissolved and free gases analysis"
    (Tabela 1 e Anexo B).
  - M. Duval, "A review of faults detectable by gas-in-oil analysis in
    transformers", IEEE Electrical Insulation Magazine, 18(3), 2002.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED  # noqa: E402

logger = logging.getLogger(__name__)

CLASSES_FALHA_6 = ["PD", "D1", "D2", "T1", "T2", "T3"]
CLASSES_7 = ["Normal"] + CLASSES_FALHA_6
# Rótulos de saída que NÃO pertencem às classes do dataset
ROTULO_NAO_CLASSIFICADO = "ND"   # sem faixa correspondente (razões)
ROTULO_ZONA_MISTA = "DT"         # zona mista térmica+elétrica (Duval)


# ---------------------------------------------------------------------------
# 1. Método das razões da IEC 60599:2015 (Tabela 1)
# ---------------------------------------------------------------------------

def _razao(num: float, den: float) -> float:
    """num/den com tratamento explícito de zeros.

    den > 0          → num/den
    den == 0, num>0  → +inf (razão indefinida, mas "muito grande")
    den == 0, num==0 → nan  (sem informação; nenhuma faixa casa)
    """
    if den > 0:
        return num / den
    return np.inf if num > 0 else np.nan


def classificar_iec_razoes(h2: float, ch4: float, c2h2: float,
                           c2h4: float, c2h6: float) -> str:
    """Classifica pela Tabela 1 da IEC 60599:2015 (método das razões).

    ┌──────┬────────────┬───────────┬────────────┐
    │      │ C2H2/C2H4  │ CH4/H2    │ C2H4/C2H6  │
    ├──────┼────────────┼───────────┼────────────┤
    │ PD   │ NS         │ < 0,1     │ < 0,2      │
    │ D1   │ > 1        │ 0,1 – 0,5 │ > 1        │
    │ D2   │ 0,6 – 2,5  │ 0,1 – 1   │ > 2        │
    │ T1   │ NS         │ > 1 (NS)  │ < 1        │
    │ T2   │ < 0,1      │ > 1       │ 1 – 4      │
    │ T3   │ < 0,2      │ > 1       │ > 4        │
    └──────┴────────────┴───────────┴────────────┘
    NS = não significativo (qualquer valor). As faixas de D1 e D2 se sobrepõem
    em parte; usa-se a ordem PD, D1, D2, T1, T2, T3 (primeira que casar).

    Returns
    -------
    str
        'PD','D1','D2','T1','T2','T3' ou 'ND' (nenhuma faixa casou).
    """
    r1 = _razao(c2h2, c2h4)   # C2H2/C2H4
    r2 = _razao(ch4, h2)      # CH4/H2
    r3 = _razao(c2h4, c2h6)   # C2H4/C2H6

    # Comparações com nan são falsas → "ND" automático quando a razão
    # necessária não está definida.
    if r2 < 0.1 and r3 < 0.2:
        return "PD"
    if r1 > 1.0 and 0.1 <= r2 <= 0.5 and r3 > 1.0:
        return "D1"
    if 0.6 <= r1 <= 2.5 and 0.1 <= r2 <= 1.0 and r3 > 2.0:
        return "D2"
    if r2 > 1.0 and r3 < 1.0:
        return "T1"
    if r1 < 0.1 and r2 > 1.0 and 1.0 <= r3 <= 4.0:
        return "T2"
    if r1 < 0.2 and r2 > 1.0 and r3 > 4.0:
        return "T3"
    return ROTULO_NAO_CLASSIFICADO


# ---------------------------------------------------------------------------
# 2. Triângulo 1 de Duval (IEC 60599:2015, Anexo B)
# ---------------------------------------------------------------------------

def _percentuais_duval(ch4: float, c2h4: float, c2h2: float) -> tuple[float, float, float]:
    """(%CH4, %C2H4, %C2H2) relativos à soma CH4+C2H4+C2H2."""
    total = ch4 + c2h4 + c2h2
    if total <= 0:
        return np.nan, np.nan, np.nan
    return 100.0 * ch4 / total, 100.0 * c2h4 / total, 100.0 * c2h2 / total


def classificar_duval(ch4: float, c2h4: float, c2h2: float) -> str:
    """Triângulo 1 de Duval, zonas conforme IEC 60599:2015 Anexo B.

    PD : %CH4 ≥ 98
    T1 : %C2H2 < 4  e %C2H4 < 20
    T2 : %C2H2 < 4  e 20 ≤ %C2H4 < 50
    T3 : %C2H2 < 15 e %C2H4 ≥ 50
    D1 : %C2H2 ≥ 13 e %C2H4 < 23
    D2 : %C2H2 ≥ 13 e %C2H4 ≥ 23 e (%C2H4 < 40 ou %C2H2 ≥ 29)
    DT : demais regiões (zona mista térmica + elétrica)

    A zona DT não é uma das classes do dataset; é devolvida como 'DT' e
    contada como erro na avaliação (não é remapeada para D1/D2).
    Se CH4+C2H4+C2H2 = 0 não há ponto no triângulo → 'ND'.
    """
    pch4, pc2h4, pc2h2 = _percentuais_duval(ch4, c2h4, c2h2)
    if np.isnan(pch4):
        return ROTULO_NAO_CLASSIFICADO

    if pch4 >= 98.0:
        return "PD"
    if pc2h2 < 4.0 and pc2h4 < 20.0:
        return "T1"
    if pc2h2 < 4.0 and 20.0 <= pc2h4 < 50.0:
        return "T2"
    if pc2h2 < 15.0 and pc2h4 >= 50.0:
        return "T3"
    if pc2h2 >= 13.0 and pc2h4 < 23.0:
        return "D1"
    if pc2h2 >= 13.0 and pc2h4 >= 23.0 and (pc2h4 < 40.0 or pc2h2 >= 29.0):
        return "D2"
    return ROTULO_ZONA_MISTA


# ---------------------------------------------------------------------------
# 3. Métricas
# ---------------------------------------------------------------------------

def _metricas(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """Métricas com rótulos FIXOS (`classes`).

    Saídas fora de `classes` (ND, DT) contam como erro; não entram como
    classe adicional no F1-macro (que é a média sobre `classes`).
    """
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0,
    )
    colunas = classes + [ROTULO_NAO_CLASSIFICADO + "/" + ROTULO_ZONA_MISTA]
    cm = np.zeros((len(classes), len(colunas)), dtype=int)
    for t, pr in zip(y_true, y_pred):
        i = classes.index(t)
        j = classes.index(pr) if pr in classes else len(classes)
        cm[i, j] += 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(np.mean(f)),
        "f1_por_classe": {c: float(x) for c, x in zip(classes, f)},
        "recall_por_classe": {c: float(x) for c, x in zip(classes, r)},
        "precision_por_classe": {c: float(x) for c, x in zip(classes, p)},
        "support": {c: int(x) for c, x in zip(classes, s)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_colunas": colunas,
    }


def _avaliar_um(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mascara_falha = y_true != "Normal"
    y_t_f, y_p_f = y_true[mascara_falha], y_pred[mascara_falha]
    cobertas = np.isin(y_p_f, CLASSES_FALHA_6)
    return {
        "somente_falhas": {
            **_metricas(y_t_f, y_p_f, CLASSES_FALHA_6),
            "n_amostras": int(mascara_falha.sum()),
            "cobertura": float(cobertas.mean()),
            "acuracia_entre_classificadas": (
                float(np.mean(y_t_f[cobertas] == y_p_f[cobertas]))
                if cobertas.any() else float("nan")
            ),
            "contagem_saidas": {k: int(v) for k, v in
                                zip(*np.unique(y_p_f, return_counts=True))},
        },
        "sete_classes": {
            **_metricas(y_true, y_pred, CLASSES_7),
            "n_amostras": int(len(y_true)),
            "nota": ("Métodos não definem a classe Normal: recall de Normal = 0 "
                     "por construção. Usar 'somente_falhas' para comparar."),
        },
    }


# ---------------------------------------------------------------------------
# 4. Avaliação no conjunto de teste
# ---------------------------------------------------------------------------

def predizer_metodos_tradicionais(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Aplica os dois métodos linha a linha (gases brutos em PPM)."""
    iec = np.array([
        classificar_iec_razoes(r.H2, r.CH4, r.C2H2, r.C2H4, r.C2H6)
        for r in df.itertuples()
    ])
    duval = np.array([
        classificar_duval(r.CH4, r.C2H4, r.C2H2) for r in df.itertuples()
    ])
    return {"IEC60599": iec, "Duval": duval}


def avaliar_metodos_tradicionais(
    caminho_csv_teste: Path | None = None,
    classes_interesse: list[str] | None = None,  # mantido por compatibilidade
) -> dict:
    """Avalia IEC 60599 (razões) e Duval no teste bruto (classification_test.csv).

    Returns
    -------
    dict
        {'IEC60599': {...}, 'Duval': {...}}, cada um com 'somente_falhas'
        e 'sete_classes' (ver `_avaliar_um`) e 'predicoes'.
    """
    if caminho_csv_teste is None:
        caminho_csv_teste = DATA_PROCESSED / "classification_test.csv"
    if not caminho_csv_teste.exists():
        raise FileNotFoundError(
            f"Arquivo de teste não encontrado: '{caminho_csv_teste}'. "
            "Execute src/preprocessing/run_preprocessing.py antes."
        )

    df = pd.read_csv(caminho_csv_teste)
    y_true = df["fault_type"].values
    predicoes = predizer_metodos_tradicionais(df)

    resultados: dict = {}
    for nome, y_pred in predicoes.items():
        res = _avaliar_um(y_true, y_pred)
        res["predicoes"] = y_pred.tolist()
        resultados[nome] = res
        logger.info(
            "%s — falhas: F1-macro=%.4f acc=%.4f cobertura=%.3f | 7 classes: F1-macro=%.4f",
            nome,
            res["somente_falhas"]["f1_macro"],
            res["somente_falhas"]["accuracy"],
            res["somente_falhas"]["cobertura"],
            res["sete_classes"]["f1_macro"],
        )
    return resultados
