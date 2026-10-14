"""
Codificação e decodificação do cromossomo híbrido para o IGA.

Estrutura do cromossomo (numpy array float64):
  cromossomo[:n_features]  → parte binária: 1.0 = feature ativa, 0.0 = desativada
  cromossomo[n_features:]  → parte contínua: 9 valores em [0.0, 1.0]
                             representando hiperparâmetros do XGBoost normalizados

A inicialização é enviesada: razões IEC têm probabilidade maior de serem ativadas,
incorporando conhecimento de domínio na geração da população inicial.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Espaço de busca dos hiperparâmetros do XGBoost
# ---------------------------------------------------------------------------
# Formato: (nome, limite_inferior, limite_superior, tipo)
HPARAM_SPECS: list[tuple] = [
    ("learning_rate",    0.01, 0.30, "float"),
    ("max_depth",        3,    12,   "int"),
    ("n_estimators",     50,   500,  "int"),
    ("subsample",        0.5,  1.0,  "float"),
    ("colsample_bytree", 0.3,  1.0,  "float"),
    ("gamma",            0.0,  5.0,  "float"),
    ("min_child_weight", 1,    10,   "int"),
    ("reg_alpha",        0.0,  1.0,  "float"),
    ("reg_lambda",       0.0,  5.0,  "float"),
]

N_HPARAMS: int = len(HPARAM_SPECS)

# ---------------------------------------------------------------------------
# Definição das features por track (ordem fixa — corresponde às colunas dos CSVs)
# ---------------------------------------------------------------------------

FEATURES_TRACK_A: list[str] = [
    # Grupo 1 — Snapshot dos gases observados (Katser)
    "obs_H2", "obs_CO", "obs_C2H4", "obs_C2H2",
    # Grupo 2 — Razões físicas calculáveis com 4 gases do Katser + TH
    "ratio_C2H2_C2H4", "ratio_H2_C2H2", 
    "TH_katser", "ratio_C2H2_TH_katser", "ratio_C2H4_TH_katser",
    # Grupo 3 — Previsões LSTM (h=14 passos → 7 dias)
    "lstm_h14_H2", "lstm_h14_CO", "lstm_h14_C2H4", "lstm_h14_C2H2",
    # Grupo 3 — Previsões LSTM (h=28 passos → 14 dias)
    "lstm_h28_H2", "lstm_h28_CO", "lstm_h28_C2H4", "lstm_h28_C2H2",
]

# Fase 5.5 — ABLAÇÃO: Track A sem nenhuma feature derivada da LSTM.
# Mesmas colunas e mesma ordem relativa do FEATURES_TRACK_A, apenas sem o Grupo 3.
FEATURES_TRACK_A_SEM_LSTM: list[str] = [
    f for f in FEATURES_TRACK_A if not f.startswith("lstm_")
]

FEATURES_TRACK_B: list[str] = [
    # Grupo 1 — Snapshot dos 5 gases (base chinesa)
    "H2", "CH4", "C2H6", "C2H4", "C2H2",
    # Grupo 2 — Razões (IEC + Doernenburg + Wu et al.)
    "ratio_CH4_H2", "ratio_C2H2_C2H4", "ratio_C2H4_C2H6", "ratio_C2H2_CH4", "ratio_C2H2_C2H6",
    # Grupo 2B — Somas e percentuais (Wu et al. Tabela 1)
    "TH", "TH1",
    "ratio_CH4_TH", "ratio_C2H4_TH", "ratio_C2H2_TH",
    "ratio_CH4_TH1", "ratio_C2H6_TH1", "ratio_C2H4_TH1", "ratio_C2H2_TH1",
]

FEATURES_TRACK_WU: list[str] = [
    "H2", "CH4", "C2H6", "C2H4", "C2H2",
    "ratio_CH4_H2", "ratio_C2H4_C2H6", "ratio_C2H2_C2H4",
    "ratio_C2H2_C2H6",
    "TH", "TH1",
    "ratio_CH4_TH", "ratio_C2H4_TH", "ratio_C2H2_TH",
    "ratio_CH4_TH1", "ratio_C2H6_TH1", "ratio_C2H4_TH1", "ratio_C2H2_TH1",
]

# Features de razão IEC — recebem probabilidade maior de ativação na inicialização
_RAZOES_IEC_A: frozenset[str] = frozenset({
    "ratio_C2H2_C2H4", "ratio_H2_C2H2", 
    "ratio_C2H2_TH_katser", "ratio_C2H4_TH_katser"
})
_RAZOES_IEC_B: frozenset[str] = frozenset({
    "ratio_CH4_H2", "ratio_C2H2_C2H4", "ratio_C2H4_C2H6", "ratio_C2H2_CH4", "ratio_C2H2_C2H6",
    "ratio_CH4_TH", "ratio_C2H4_TH", "ratio_C2H2_TH",
    "ratio_CH4_TH1", "ratio_C2H6_TH1", "ratio_C2H4_TH1", "ratio_C2H2_TH1"
})
_RAZOES_IEC_WU: frozenset[str] = frozenset({
    "ratio_CH4_H2", "ratio_C2H4_C2H6", "ratio_C2H2_C2H4", "ratio_C2H2_C2H6",
    "ratio_CH4_TH", "ratio_C2H4_TH", "ratio_C2H2_TH",
    "ratio_CH4_TH1", "ratio_C2H6_TH1", "ratio_C2H4_TH1", "ratio_C2H2_TH1",
})

MIN_FEATURES_ATIVAS: int = 2  # restrição de viabilidade


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _specs_do_track(track: str) -> tuple[list[str], frozenset[str]]:
    """Retorna (lista_de_features, set_de_razoes_iec) para o track informado."""
    if track == "A":
        return FEATURES_TRACK_A, _RAZOES_IEC_A
    if track == "A_SEM_LSTM":
        return FEATURES_TRACK_A_SEM_LSTM, _RAZOES_IEC_A
    if track == "B":
        return FEATURES_TRACK_B, _RAZOES_IEC_B
    if track == "WU":
        return FEATURES_TRACK_WU, _RAZOES_IEC_WU
    raise ValueError(f"Track inválido: '{track}'. Use 'A', 'A_SEM_LSTM', 'B' ou 'WU'.")


def cromossomo_len(track: str) -> int:
    """Tamanho total do cromossomo para o track dado."""
    features, _ = _specs_do_track(track)
    return len(features) + N_HPARAMS


# ---------------------------------------------------------------------------
# Geração de cromossomo
# ---------------------------------------------------------------------------

def gerar_cromossomo(
    track: str,
    rng: np.random.Generator,
    prob_iec: float = 0.8,
    prob_demais: float = 0.5,
) -> np.ndarray:
    """Gera um cromossomo aleatório com inicialização enviesada por domínio.

    Razões IEC têm probabilidade `prob_iec` de serem ativadas; demais
    features têm probabilidade `prob_demais`. Garante ao menos
    MIN_FEATURES_ATIVAS features ativas.

    Parameters
    ----------
    track : str
        'A' ou 'B'.
    rng : np.random.Generator
        Gerador de números aleatórios (para reprodutibilidade).
    prob_iec : float
        Probabilidade de ativar features de razão IEC (padrão: 0.8).
    prob_demais : float
        Probabilidade de ativar demais features (padrão: 0.5).

    Returns
    -------
    cromossomo : np.ndarray
        Array de shape (n_features + N_HPARAMS,) com dtype float64.
    """
    features, razoes_iec = _specs_do_track(track)
    n_features = len(features)

    # Parte binária: inicialização enviesada
    bits = np.array([
        1.0 if rng.random() < (prob_iec if f in razoes_iec else prob_demais) else 0.0
        for f in features
    ], dtype=np.float64)

    # Garante o mínimo de features ativas
    if bits.sum() < MIN_FEATURES_ATIVAS:
        idx = rng.choice(n_features, size=MIN_FEATURES_ATIVAS, replace=False)
        bits[idx] = 1.0

    # Parte contínua: uniforme [0, 1]
    continuo = rng.uniform(0.0, 1.0, size=N_HPARAMS).astype(np.float64)

    return np.concatenate([bits, continuo])


# ---------------------------------------------------------------------------
# Decodificação
# ---------------------------------------------------------------------------

def decodificar(
    cromossomo: np.ndarray,
    track: str,
) -> tuple[list[str], dict]:
    """Decodifica o cromossomo em features ativas e hiperparâmetros do XGBoost.

    Parameters
    ----------
    cromossomo : np.ndarray
        Cromossomo híbrido.
    track : str
        'A' ou 'B'.

    Returns
    -------
    features_ativas : list[str]
        Nomes das features com bit ≥ 0.5 (ativas).
    hparams : dict
        Hiperparâmetros do XGBoost em suas escalas reais.
    """
    features, _ = _specs_do_track(track)
    n_features = len(features)

    bits = cromossomo[:n_features]
    continuo = cromossomo[n_features:]

    features_ativas = [f for f, b in zip(features, bits) if b >= 0.5]

    hparams: dict = {}
    for i, (nome, lo, hi, tipo) in enumerate(HPARAM_SPECS):
        val = lo + float(continuo[i]) * (hi - lo)
        hparams[nome] = int(round(val)) if tipo == "int" else float(val)

    return features_ativas, hparams


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

def validar(cromossomo: np.ndarray, track: str) -> bool:
    """Verifica se o cromossomo possui pelo menos MIN_FEATURES_ATIVAS ativas."""
    features, _ = _specs_do_track(track)
    n_features = len(features)
    return int(cromossomo[:n_features].sum()) >= MIN_FEATURES_ATIVAS
