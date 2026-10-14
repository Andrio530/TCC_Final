"""
Fase 5.5 — utilidades estatísticas: bootstrap estratificado (pareado).

Por que existe: o "± desvio" sobre 5 seeds mede apenas a aleatoriedade do
XGBoost; NÃO mede a incerteza amostral de um conjunto de teste pequeno
(295 amostras na base chinesa, 16 em T1/T2). O bootstrap sobre as amostras de
teste mede essa incerteza.

Desenho:
  - Bootstrap ESTRATIFICADO: reamostra com reposição DENTRO de cada classe,
    preservando o suporte por classe. Evita o artefato de uma réplica sem
    exemplos de uma classe (que zeraria artificialmente o F1 dela).
    Consequência: a incerteza é condicional às contagens por classe do teste.
  - PAREADO: todos os modelos são avaliados nos MESMOS índices reamostrados,
    de modo que a diferença entre modelos tem IC próprio (muito mais estreito
    que comparar dois ICs individuais).
  - Cada modelo é representado por predições de várias seeds (matriz S × n).
    A métrica por réplica é a MÉDIA do F1-macro sobre as seeds — idêntica à
    métrica "multiseed" reportada nos JSONs da Fase 5.
"""

from __future__ import annotations

import numpy as np


def f1_macro_rapido(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """F1-macro (zero_division=0) sobre rótulos inteiros 0..n_classes-1.

    `y_pred` pode conter o valor `n_classes`, que significa "fora das classes
    avaliadas" (ex.: ND/DT nas regras, ou 'Normal' na visão só-falhas). Essas
    predições contam como erro (FN da classe real) mas NÃO formam uma classe
    extra na média do F1-macro.
    """
    n_cols = n_classes + 1
    idx = y_true * n_cols + y_pred
    cm = np.bincount(idx, minlength=n_classes * n_cols).reshape(n_classes, n_cols)
    tp = np.diag(cm[:, :n_classes]).astype(float)
    fp = cm[:, :n_classes].sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def indices_bootstrap_estratificado(
    y: np.ndarray, n_boot: int, rng: np.random.Generator,
) -> np.ndarray:
    """Matriz (n_boot, n) de índices reamostrados dentro de cada classe."""
    n = len(y)
    saida = np.empty((n_boot, n), dtype=np.int64)
    posicoes_por_classe = [np.where(y == c)[0] for c in np.unique(y)]
    col = 0
    for pos in posicoes_por_classe:
        k = len(pos)
        amostra = rng.integers(0, k, size=(n_boot, k))
        saida[:, col:col + k] = pos[amostra]
        col += k
    return saida


def _f1_medio_sobre_seeds(
    y: np.ndarray, preds: np.ndarray, idx: np.ndarray, n_classes: int,
) -> float:
    return float(np.mean([
        f1_macro_rapido(y[idx], p[idx], n_classes) for p in preds
    ]))


def bootstrap_modelos(
    y: np.ndarray,
    preds_por_modelo: dict[str, np.ndarray],
    n_classes: int,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap estratificado e pareado.

    Parameters
    ----------
    y : (n,) rótulos inteiros reais.
    preds_por_modelo : nome → (S, n) predições inteiras (S seeds; use S=1 para
        modelos determinísticos como as regras Duval/IEC).
    n_classes : número de classes (rótulos 0..n_classes-1).

    Returns
    -------
    dict com:
      'modelos'    : nome → {f1_ponto, ic_inf, ic_sup, std_boot}
      'dist'       : nome → np.ndarray (n_boot,)  [para comparações]
    """
    rng = np.random.default_rng(seed)
    indices = indices_bootstrap_estratificado(y, n_boot, rng)
    ponto_idx = np.arange(len(y))

    resultado: dict = {"modelos": {}, "dist": {}, "n_boot": n_boot, "alpha": alpha}
    for nome, preds in preds_por_modelo.items():
        preds = np.atleast_2d(preds).astype(np.int64)
        ponto = _f1_medio_sobre_seeds(y, preds, ponto_idx, n_classes)
        dist = np.array([
            _f1_medio_sobre_seeds(y, preds, idx, n_classes) for idx in indices
        ])
        lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
        resultado["modelos"][nome] = {
            "f1_ponto": ponto,
            "ic_inf": float(lo),
            "ic_sup": float(hi),
            "std_boot": float(dist.std(ddof=1)),
        }
        resultado["dist"][nome] = dist
    return resultado


def comparar_pareado(
    res_boot: dict, a: str, b: str, alpha: float = 0.05,
) -> dict:
    """Diferença F1(a) − F1(b) sobre as MESMAS réplicas bootstrap.

    p_valor é bicaudal e aproximado: 2·min(P(d ≤ 0), P(d ≥ 0)), com piso
    1/n_boot. Interprete junto com o IC; NÃO é um teste de hipótese exato
    (o bootstrap condiciona no conjunto de teste observado).
    """
    d = res_boot["dist"][a] - res_boot["dist"][b]
    n = len(d)
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    p = float(min(1.0, max(p, 1.0 / n)))
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    ponto = res_boot["modelos"][a]["f1_ponto"] - res_boot["modelos"][b]["f1_ponto"]
    return {
        "a": a,
        "b": b,
        "delta_f1_ponto": float(ponto),
        "ic_inf": float(lo),
        "ic_sup": float(hi),
        "p_valor_aprox": p,
        "ic_exclui_zero": bool(lo > 0 or hi < 0),
    }


def teste_t_corrigido_nadeau_bengio(
    diffs: np.ndarray, n_treino: int, n_teste: int,
) -> dict:
    """t reamostrado corrigido (Nadeau & Bengio, 2003) para validação cruzada
    repetida. A correção (1/k + n_teste/n_treino) reconhece que as estimativas
    por fold NÃO são independentes (os conjuntos de treino se sobrepõem).
    """
    from scipy import stats  # import tardio: só necessário aqui

    diffs = np.asarray(diffs, dtype=float)
    k = len(diffs)
    media = float(diffs.mean())
    var = float(diffs.var(ddof=1))
    se = np.sqrt((1.0 / k + n_teste / n_treino) * var) if var > 0 else 0.0
    if se == 0:
        return {"media": media, "t": float("nan"), "p_valor": float("nan"),
                "ic_inf": media, "ic_sup": media, "k": k}
    t = media / se
    p = float(2 * stats.t.sf(abs(t), df=k - 1))
    crit = stats.t.ppf(0.975, df=k - 1)
    return {"media": media, "t": float(t), "p_valor": p,
            "ic_inf": float(media - crit * se),
            "ic_sup": float(media + crit * se), "k": k}
