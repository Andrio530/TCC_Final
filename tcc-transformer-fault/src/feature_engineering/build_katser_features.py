"""
Fase 4 — Engenharia de Atributos (Track A: base Katser).

Para cada série Katser (treino / val / teste), constrói o vetor de atributos
que alimentará o XGBoost no Estágio 2 do sistema híbrido:

  Grupo 1 — Snapshot: últimos valores observados dos 4 gases
             (H2, CO, C2H4, C2H2) no instante de diagnóstico.

  Grupo 2 — Razões IEC: razões físicas calculáveis a partir dos gases Katser.
             • C2H2/C2H4  → principal discriminador elétrico/térmico (IEC 60599)
             • H2/C2H2    → sensibilidade a descargas parciais
             • CO/C2H4    → degradação de celulose vs. pirólise do óleo

  Grupo 3 — Previsão LSTM: concentrações futuras previstas pela LSTM treinada
             na Fase 3 para os horizontes definidos em HORIZONTES_H.
             A janela de entrada são os ÚLTIMOS w*=50 pontos REAIS da série
             (nenhum dado sintético é gerado).

Saídas (data/processed/):
  katser_features_train.csv
  katser_features_val.csv
  katser_features_test.csv
  katser_feature_scaler.pkl   ← MinMaxScaler fitado apenas no treino

Uso:
  python -m src.feature_engineering.build_katser_features
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DATA_PROCESSED,
    KATSER_CLASSES,
    KATSER_GASES,
    RANDOM_SEED,
)
from src.lstm.data_loader import ScalerPerGas  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parâmetros da Fase 4
# ---------------------------------------------------------------------------

# Janela ótima encontrada na Fase 3 (w* = 50)
W_STAR = 50

# Horizontes de previsão (em passos, onde 1 passo = 12 horas)
# h=14  →  7 dias   (horizonte operacional primário)
# h=28  → 14 dias   (horizonte secundário para ablação)
HORIZONTES_H = [14, 28]

# Modelo com normalização per-gas (melhor configuração da Fase 3)
NORMALIZAR = True
SUFIXO_MODELO = "_norm"


# ---------------------------------------------------------------------------
# 1. Carregar splits Katser
# ---------------------------------------------------------------------------

def _carregar_split(nome: str) -> tuple[np.ndarray, np.ndarray]:
    """Carrega um split .npz e retorna (series, labels)."""
    caminho = DATA_PROCESSED / nome
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo '{caminho}' não encontrado. "
            "Execute src/preprocessing/run_preprocessing.py antes desta etapa."
        )
    d = np.load(caminho)
    return d["series"].astype(np.float32), d["labels"]


def carregar_splits() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Carrega treino, val e teste Katser."""
    splits = {
        "train": _carregar_split("katser_train.npz"),
        "val":   _carregar_split("katser_val.npz"),
        "test":  _carregar_split("katser_test.npz"),
    }
    for nome, (s, l) in splits.items():
        logger.info("Split %-5s → séries: %s | labels: %s", nome, s.shape, l.shape)
    return splits


# ---------------------------------------------------------------------------
# 2. Carregar LSTM treinada + scaler da Fase 3
# ---------------------------------------------------------------------------

def _carregar_lstm_e_scaler() -> tuple:
    """Carrega o modelo Keras e o ScalerPerGas usados na Fase 3."""
    import tensorflow as tf  # importação lazy para não impactar imports sem GPU

    cfg_path = DATA_PROCESSED / f"lstm_best_config{SUFIXO_MODELO}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Configuração LSTM não encontrada em '{cfg_path}'. "
            "Execute src/lstm/run_lstm.py antes desta etapa."
        )
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    model_path = DATA_PROCESSED / f"lstm_best_model{SUFIXO_MODELO}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo LSTM não encontrado em '{model_path}'.")

    logger.info("Carregando modelo LSTM de '%s'...", model_path)
    modelo = tf.keras.models.load_model(model_path)
    logger.info("Modelo carregado. Configuração: %s", cfg)

    return modelo, cfg


def _reconstruir_scaler(series_treino_raw: np.ndarray) -> ScalerPerGas:
    """Refaz o fit do ScalerPerGas nos dados de treino brutos (reprodutível)."""
    scaler = ScalerPerGas().fit(series_treino_raw)
    logger.info("ScalerPerGas refitado nos dados de treino.")
    return scaler


# ---------------------------------------------------------------------------
# 3. Previsão recursiva da LSTM (Grupo 3)
# ---------------------------------------------------------------------------

def prever_gases_futuros(
    modelo,
    janela_norm: np.ndarray,
    h: int,
) -> np.ndarray:
    """Previsão recursiva: usa a LSTM h vezes para olhar h passos à frente.

    A cada iteração, a previsão do passo anterior é adicionada à janela e o
    passo mais antigo é descartado (rolling window). Isso é a estratégia
    multi-step mais consistente com o treinamento one-step-ahead da LSTM.

    Parameters
    ----------
    modelo : tf.keras.Model
        LSTM treinada na Fase 3 (input: (1, W, 4) → output: (1, 4)).
    janela_norm : np.ndarray
        Janela de entrada normalizada, shape (W, 4).
    h : int
        Número de passos à frente a prever.

    Returns
    -------
    previsoes : np.ndarray
        Concentrações previstas no passo h, shape (4,).
        Retorna apenas o último passo (h-ésimo) como representante do futuro.
    """
    janela_atual = janela_norm.copy()  # (W, 4)

    for _ in range(h):
        x_input = janela_atual[np.newaxis, :, :]          # (1, W, 4)
        proximo = modelo.predict(x_input, verbose=0)[0]   # (4,)
        janela_atual = np.vstack([janela_atual[1:], proximo])  # slide da janela

    return proximo  # shape (4,) — concentrações previstas no passo h


# ---------------------------------------------------------------------------
# 4. Razões físicas — Grupo 2
# ---------------------------------------------------------------------------

def calcular_razoes(snapshot: np.ndarray) -> dict[str, float]:
    """Calcula razões físicas e features de soma dos gases Katser.

    Gases disponíveis: [H2, CO, C2H4, C2H2] → índices [0, 1, 2, 3]
    Nota: CH4 e C2H6 não estão disponíveis nesta base — TH e TH1 completos
    não podem ser calculados.

    Razões calculadas:
      C2H2/C2H4     — razão IEC 60599: discrimina descarga elétrica (alto)
                      vs. falha térmica (baixo). Fonte: IEC 60599, Rogers.
      H2/C2H2       — adaptação metodológica para esta base: discrimina PD
                      (H2 elevado, C2H2 baixo → valor alto) vs. arco de alta
                      energia (C2H2 elevado → valor baixo). Declarada como
                      adaptação pois CH4 não está disponível para usar a razão
                      padrão CH4/H2. Não é razão padronizada nos métodos
                      clássicos; incluída pela lógica física defensável.

    Feature de soma removida:
      CO/C2H4 foi removida — não possui fundamentação em nenhum método
      clássico de DGA (IEC 60599, Rogers, Doernenburg, Wu et al. 2021).
      CO e C2H4 são produzidos por mecanismos independentes (decomposição
      de celulose vs. pirólise do óleo mineral) sem relação proporcional
      definida na literatura.

    Features de soma (versão parcial de TH, adaptada aos gases disponíveis):
      TH_katser           = C2H4 + C2H2  (sem CH4 — ausente na base Katser)
      ratio_C2H2_TH_katser = C2H2 / TH_katser
      ratio_C2H4_TH_katser = C2H4 / TH_katser
    """
    H2, CO, C2H4, C2H2 = snapshot  # CO mantido no snapshot (Grupo 1), não em razões

    def razao_segura(num: float, den: float) -> float:
        """Retorna num/den; retorna 0.0 se denominador ≤ 0 (evita inf/NaN)."""
        return float(num / den) if den > 0 else 0.0

    # TH parcial — soma dos hidrocarbonetos combustíveis disponíveis
    TH_katser = C2H4 + C2H2

    return {
        # --- Razão IEC 60599 ---
        "ratio_C2H2_C2H4":      razao_segura(C2H2, C2H4),
        # --- Adaptação metodológica (declarada no TCC) ---
        "ratio_H2_C2H2":        razao_segura(H2, C2H2),
        # --- TH parcial e percentuais relativos ---
        "TH_katser":            TH_katser,
        "ratio_C2H2_TH_katser": razao_segura(C2H2, TH_katser),
        "ratio_C2H4_TH_katser": razao_segura(C2H4, TH_katser),
    }


# ---------------------------------------------------------------------------
# 5. Construção do vetor de features para um split completo
# ---------------------------------------------------------------------------

def _construir_features_split(
    series: np.ndarray,
    labels: np.ndarray,
    modelo,
    scaler: ScalerPerGas,
    split_nome: str,
) -> pd.DataFrame:
    """Constrói o DataFrame de features para um split inteiro.

    Para cada série:
      1. Extrai o snapshot (últimos valores observados) — Grupo 1.
      2. Calcula as razões físicas sobre o snapshot — Grupo 2.
      3. Normaliza a janela w* e roda a LSTM para cada horizonte h — Grupo 3.

    Parameters
    ----------
    series : np.ndarray
        Shape (n_series, T, 4).
    labels : np.ndarray
        Shape (n_series,) — inteiros 1–4 (KATSER_CLASSES).
    modelo : tf.keras.Model
        LSTM treinada (Fase 3).
    scaler : ScalerPerGas
        Scaler fitado no treino.
    split_nome : str
        Nome do split (para logs).

    Returns
    -------
    df : pd.DataFrame
        Uma linha por série, colunas = features + 'fault_type'.
    """
    n_series = len(series)
    registros = []

    for i, (serie, label) in enumerate(zip(series, labels)):
        if (i + 1) % 200 == 0 or i == n_series - 1:
            logger.info("[%s] Processando série %d/%d...", split_nome, i + 1, n_series)

        # --- Grupo 1: snapshot (últimos valores brutos da série) ---
        snapshot_raw = serie[-1, :]   # (4,) — último passo da série (em escala raw)
        grupo1 = {
            f"obs_{gas}": float(snapshot_raw[j])
            for j, gas in enumerate(KATSER_GASES)
        }

        # --- Grupo 2: razões físicas (calculadas sobre snapshot raw) ---
        grupo2 = calcular_razoes(snapshot_raw)

        # --- Grupo 3: previsões LSTM (janela normalizada) ---
        janela_raw = serie[-W_STAR:, :]   # (50, 4) — últimos w* pontos reais
        # Normaliza a janela usando o scaler fitado no treino
        janela_norm = scaler.transform(janela_raw[np.newaxis, :, :])[0]  # (50, 4)

        grupo3 = {}
        for h in HORIZONTES_H:
            previsao_norm = prever_gases_futuros(modelo, janela_norm, h=h)
            # Desnormaliza para manter as features na mesma escala do Grupo 1
            previsao_raw = scaler.inverse_transform(
                previsao_norm[np.newaxis, np.newaxis, :]
            )[0, 0, :]  # (4,)
            for j, gas in enumerate(KATSER_GASES):
                grupo3[f"lstm_h{h}_{gas}"] = float(previsao_raw[j])

        # --- Label ---
        label_str = KATSER_CLASSES[int(label)]

        registros.append({**grupo1, **grupo2, **grupo3, "fault_type": label_str})

    df = pd.DataFrame(registros)
    logger.info(
        "[%s] DataFrame construído: %d linhas × %d colunas.",
        split_nome, len(df), df.shape[1],
    )
    return df


# ---------------------------------------------------------------------------
# 6. Normalização do vetor final (MinMaxScaler fitado no treino)
# ---------------------------------------------------------------------------

def normalizar_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplica MinMaxScaler às colunas numéricas, fitado apenas no treino."""
    from sklearn.preprocessing import MinMaxScaler  # noqa: PLC0415

    colunas_features = [c for c in df_train.columns if c != "fault_type"]
    scaler_final = MinMaxScaler()

    df_train_out = df_train.copy()
    df_val_out   = df_val.copy()
    df_test_out  = df_test.copy()

    df_train_out[colunas_features] = scaler_final.fit_transform(
        df_train[colunas_features]
    )
    df_val_out[colunas_features]  = scaler_final.transform(df_val[colunas_features])
    df_test_out[colunas_features] = scaler_final.transform(df_test[colunas_features])

    # Persiste o scaler para uso na inferência futura
    scaler_path = DATA_PROCESSED / "katser_feature_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_final, f)
    logger.info("MinMaxScaler final salvo em '%s'.", scaler_path)

    return df_train_out, df_val_out, df_test_out


# ---------------------------------------------------------------------------
# 7. Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(RANDOM_SEED)
    logger.info("=" * 60)
    logger.info("Fase 4A — Engenharia de Atributos (Track A: Katser)")
    logger.info("Gases: %s | w*=%d | Horizontes: %s", KATSER_GASES, W_STAR, HORIZONTES_H)
    logger.info("=" * 60)

    # 1. Carrega splits
    splits = carregar_splits()
    s_train, l_train = splits["train"]
    s_val,   l_val   = splits["val"]
    s_test,  l_test  = splits["test"]

    # 2. Carrega LSTM e reconstrói scaler da Fase 3
    modelo, _ = _carregar_lstm_e_scaler()
    scaler_lstm = _reconstruir_scaler(s_train)

    # 3. Constrói features para cada split
    logger.info("Construindo features — treino...")
    df_train = _construir_features_split(s_train, l_train, modelo, scaler_lstm, "train")

    logger.info("Construindo features — validação...")
    df_val   = _construir_features_split(s_val,   l_val,   modelo, scaler_lstm, "val")

    logger.info("Construindo features — teste...")
    df_test  = _construir_features_split(s_test,  l_test,  modelo, scaler_lstm, "test")

    # 4. Normalização final (fitada no treino)
    logger.info("Normalizando features (MinMaxScaler fitado no treino)...")
    df_train, df_val, df_test = normalizar_features(df_train, df_val, df_test)

    # 5. Persiste os CSVs
    out_train = DATA_PROCESSED / "katser_features_train.csv"
    out_val   = DATA_PROCESSED / "katser_features_val.csv"
    out_test  = DATA_PROCESSED / "katser_features_test.csv"

    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val,     index=False)
    df_test.to_csv(out_test,   index=False)

    logger.info("CSVs salvos:")
    logger.info("  %s (%d linhas)", out_train.name, len(df_train))
    logger.info("  %s (%d linhas)", out_val.name,   len(df_val))
    logger.info("  %s (%d linhas)", out_test.name,  len(df_test))

    # 6. Relatório resumido
    logger.info("-" * 60)
    logger.info("Distribuição de classes (treino):")
    for classe, contagem in df_train["fault_type"].value_counts().items():
        logger.info("  %-8s → %d amostras (%.1f%%)", classe, contagem,
                    100 * contagem / len(df_train))

    logger.info("Colunas geradas: %s", list(df_train.columns))
    logger.info("=" * 60)
    logger.info("Fase 4A concluída. Arquivos prontos para a Fase 5 (IGA-XGBoost).")


if __name__ == "__main__":
    main()
