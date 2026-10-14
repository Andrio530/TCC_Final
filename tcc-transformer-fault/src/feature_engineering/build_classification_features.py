"""
Fase 4 — Engenharia de Atributos (Track B: base de classificação chinesa).

Para cada amostra da base chinesa (treino / val / teste), constrói o vetor
de atributos que alimentará o XGBoost no Estágio 2 do sistema híbrido:

  Grupo 1 — Snapshot: concentrações brutas dos 5 gases disponíveis
             (H2, CH4, C2H6, C2H4, C2H2) em PPM.

  Grupo 2 — Razões IEC 60599: as 4 razões canônicas completamente
             calculáveis a partir dos 5 gases da base chinesa.
             • CH4/H2    → discrimina PD vs. sobrecarga térmica
             • C2H2/C2H4 → discrimina descargas elétricas vs. falhas térmicas
             • C2H4/C2H6 → indica nível de temperatura da falha térmica
             • C2H2/CH4  → razão auxiliar para descargas de alta energia

  Grupo 3 — Não aplicável: a base chinesa contém snapshots estáticos,
             sem séries temporais associadas. A ausência do Grupo 3 é
             declarada como limitação metodológica no texto do TCC.

Saídas (data/processed/):
  china_features_train.csv       ← treino sem SMOTE (para análise pura)
  china_features_train_smote.csv ← treino com SMOTE (para treinar o XGBoost)
  china_features_val.csv
  china_features_test.csv
  china_feature_scaler.pkl       ← pipeline log1p + MinMaxScaler (fit no treino)

Uso:
  python -m src.feature_engineering.build_classification_features
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CLASSIFICATION_GASES,
    DATA_PROCESSED,
    RANDOM_SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Ordem dos gases: [H2, CH4, C2H6, C2H4, C2H2] → índices [0, 1, 2, 3, 4]
_IDX = {gas: i for i, gas in enumerate(CLASSIFICATION_GASES)}

# Razões de gases — base de classificação (5 gases disponíveis)
# Fontes: IEC 60599, Rogers (1975), Doernenburg (IEEE), Wu et al. (2021).
# Formato: (nome_coluna, numerador_gas, denominador_gas)
RAZOES_GASES = [
    # --- IEC 60599 / Rogers ---
    ("ratio_CH4_H2",    "CH4",  "H2"),
    ("ratio_C2H2_C2H4", "C2H2", "C2H4"),
    ("ratio_C2H4_C2H6", "C2H4", "C2H6"),
    # --- Doernenburg ---
    ("ratio_C2H2_CH4",  "C2H2", "CH4"),
    # --- Wu et al. (2021) Tabela 1 — inverso do Doernenburg C2H6/C2H2 ---
    ("ratio_C2H2_C2H6", "C2H2", "C2H6"),
]


# ---------------------------------------------------------------------------
# 1. Carga dos CSVs gerados na Fase 2 (pré-processamento)
# ---------------------------------------------------------------------------

def _carregar_csv(nome: str) -> pd.DataFrame:
    """Carrega um CSV de data/processed/ e valida colunas mínimas."""
    caminho = DATA_PROCESSED / nome
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo '{caminho}' não encontrado. "
            "Execute src/preprocessing/run_preprocessing.py antes desta etapa."
        )
    df = pd.read_csv(caminho)
    colunas_esperadas = CLASSIFICATION_GASES + ["fault_type"]
    ausentes = [c for c in colunas_esperadas if c not in df.columns]
    if ausentes:
        raise ValueError(f"Colunas ausentes em '{nome}': {ausentes}")
    return df


def carregar_splits() -> dict[str, pd.DataFrame]:
    """Carrega os quatro CSVs da base de classificação (Fase 2)."""
    splits = {
        "train":       _carregar_csv("classification_train.csv"),
        "train_smote": _carregar_csv("classification_train_smote.csv"),
        "val":         _carregar_csv("classification_val.csv"),
        "test":        _carregar_csv("classification_test.csv"),
    }
    for nome, df in splits.items():
        logger.info(
            "Split %-12s → %d amostras | classes: %s",
            nome, len(df),
            df["fault_type"].value_counts().to_dict(),
        )
    return splits


# ---------------------------------------------------------------------------
# 2. Grupo 2 — Razões de gases (IEC 60599 + Doernenburg + Wu et al.)
# ---------------------------------------------------------------------------

def calcular_razoes_gases(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula as 5 razões de gases e retorna DataFrame com colunas novas.

    Fontes: IEC 60599, Rogers (1975), Doernenburg (IEEE), Wu et al. (2021).
    Divisão segura: se o denominador for 0 (ou NaN), a razão vale 0.0.
    Não altera os valores originais dos gases (operação sobre cópia).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas dos 5 gases em PPM.

    Returns
    -------
    df_razoes : pd.DataFrame
        DataFrame apenas com as colunas de razões (sem gases brutos).
    """
    razoes = {}
    for nome_col, num_gas, den_gas in RAZOES_GASES:
        num = df[num_gas].values.astype(float)
        den = df[den_gas].values.astype(float)
        # Evita inf e NaN: razão = 0 quando denominador ≤ 0
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 0, num / den, 0.0)
        razoes[nome_col] = r
    return pd.DataFrame(razoes, index=df.index)


# ---------------------------------------------------------------------------
# 3. Grupo 2B — Features TH / TH1 (Wu et al. 2021, Tabela 1)
# ---------------------------------------------------------------------------

def calcular_th_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula TH, TH1 e seus percentuais relativos (Wu et al. 2021, Tabela 1).

    TH  = CH4 + C2H4 + C2H2        — gases do Triângulo de Duval
    TH1 = CH4 + C2H4 + C2H2 + C2H6 — todos os hidrocarbonetos combustíveis

    Os percentuais X/TH e X/TH1 são equivalentes às coordenadas normalizadas
    do Triângulo de Duval (CH4/TH, C2H4/TH, C2H2/TH). Quando TH ou TH1 é
    zero, os percentuais valem 0.0 (divisão segura).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas dos 5 gases em PPM.

    Returns
    -------
    df_th : pd.DataFrame
        DataFrame com TH, TH1 e os 7 percentuais relativos.
    """
    th_features = {}

    # --- Somas ---
    TH  = df["CH4"].values.astype(float) + df["C2H4"].values.astype(float) + df["C2H2"].values.astype(float)
    TH1 = TH + df["C2H6"].values.astype(float)

    th_features["TH"]  = TH
    th_features["TH1"] = TH1

    # --- Percentuais relativos a TH (Triângulo de Duval) ---
    for gas in ["CH4", "C2H4", "C2H2"]:
        col = f"ratio_{gas}_TH"
        num = df[gas].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            th_features[col] = np.where(TH > 0, num / TH, 0.0)

    # --- Percentuais relativos a TH1 ---
    for gas in ["CH4", "C2H6", "C2H4", "C2H2"]:
        col = f"ratio_{gas}_TH1"
        num = df[gas].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            th_features[col] = np.where(TH1 > 0, num / TH1, 0.0)

    return pd.DataFrame(th_features, index=df.index)


# ---------------------------------------------------------------------------
# 4. Montagem do vetor de features (Grupos 1 + 2 + 2B)
# ---------------------------------------------------------------------------

def montar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combina Grupo 1 (gases brutos), Grupo 2 (razões) e Grupo 2B (TH/TH1).

    Grupo 1  — 5 concentrações brutas (H2, CH4, C2H6, C2H4, C2H2)
    Grupo 2  — 5 razões: IEC 60599 + Doernenburg + Wu et al. (2021)
    Grupo 2B — 9 features TH/TH1: 2 somas + 7 percentuais relativos
    Total    — 19 features + label fault_type

    O label 'fault_type' é mantido como última coluna.
    """
    grupo1  = df[CLASSIFICATION_GASES].copy()
    grupo2  = calcular_razoes_gases(df)
    grupo2b = calcular_th_features(df)
    label   = df[["fault_type"]].copy()
    return pd.concat([grupo1, grupo2, grupo2b, label], axis=1)


# ---------------------------------------------------------------------------
# 4. Normalização: pipeline log1p → MinMaxScaler (fit apenas no treino)
# ---------------------------------------------------------------------------

def _construir_pipeline_normalizacao() -> Pipeline:
    """Cria o pipeline sklearn: log1p → MinMaxScaler.

    O log1p é aplicado antes do scaler porque as concentrações de gases
    têm distribuições fortemente assimétricas à direita (lognormal).
    Aplicar log1p comprime os valores extremos e aproxima a distribuição
    de uma gaussiana, melhorando o aprendizado do XGBoost.
    """
    return Pipeline([
        ("log1p", FunctionTransformer(np.log1p, validate=True)),
        ("scaler", MinMaxScaler()),
    ])


def normalizar_splits(
    splits_raw: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Aplica o pipeline de normalização, fitado apenas no treino sem SMOTE.

    O fit é feito no treino SEM SMOTE porque os dados sintéticos do SMOTE
    não devem influenciar os parâmetros de normalização (min/max globais).

    Parameters
    ----------
    splits_raw : dict
        Dicionário com DataFrames de features+label (ANTES da normalização).
        Chaves esperadas: 'train', 'train_smote', 'val', 'test'.

    Returns
    -------
    splits_norm : dict
        Mesmo dicionário com features normalizadas, labels inalterados.
    pipeline : Pipeline
        Pipeline fitado (salvo para uso na inferência).
    """
    colunas_features = [c for c in splits_raw["train"].columns if c != "fault_type"]

    pipeline = _construir_pipeline_normalizacao()
    pipeline.fit(splits_raw["train"][colunas_features])
    logger.info(
        "Pipeline de normalização fitado no treino (%d amostras, %d features).",
        len(splits_raw["train"]), len(colunas_features),
    )

    splits_norm = {}
    for nome, df in splits_raw.items():
        df_out = df.copy()
        df_out[colunas_features] = pipeline.transform(df[colunas_features])
        splits_norm[nome] = df_out
        logger.info(
            "Normalização aplicada em '%s': shape %s.", nome, df_out.shape
        )

    return splits_norm, pipeline


# ---------------------------------------------------------------------------
# 5. Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(RANDOM_SEED)
    logger.info("=" * 60)
    logger.info("Fase 4B — Engenharia de Atributos (Track B: China)")
    logger.info("Gases: %s", CLASSIFICATION_GASES)
    logger.info("Razões de gases (%d): %s", len(RAZOES_GASES), [r[0] for r in RAZOES_GASES])
    logger.info("Features TH/TH1: TH, TH1 + 7 percentuais relativos (Wu et al. 2021)")
    logger.info("Total features candidatas: %d", len(CLASSIFICATION_GASES) + len(RAZOES_GASES) + 9)
    logger.info("=" * 60)

    # 1. Carrega splits da Fase 2
    splits_brutos = carregar_splits()

    # 2. Monta features (Grupos 1 + 2) para cada split
    splits_features: dict[str, pd.DataFrame] = {}
    for nome, df in splits_brutos.items():
        splits_features[nome] = montar_features(df)
        logger.info(
            "Features montadas para '%s': %d linhas × %d colunas.",
            nome, len(splits_features[nome]), splits_features[nome].shape[1],
        )

    # 3. Normalização (log1p + MinMax), fitada no treino sem SMOTE
    splits_norm, pipeline = normalizar_splits(splits_features)

    # 4. Persiste o pipeline para uso na inferência futura
    scaler_path = DATA_PROCESSED / "china_feature_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Pipeline de normalização salvo em '%s'.", scaler_path)

    # 5. Salva os CSVs finais
    mapa_saida = {
        "train":       "china_features_train.csv",
        "train_smote": "china_features_train_smote.csv",
        "val":         "china_features_val.csv",
        "test":        "china_features_test.csv",
    }
    for chave, nome_arquivo in mapa_saida.items():
        destino = DATA_PROCESSED / nome_arquivo
        splits_norm[chave].to_csv(destino, index=False)
        logger.info("Salvo: %s (%d linhas)", nome_arquivo, len(splits_norm[chave]))

    # 6. Relatório resumido
    logger.info("-" * 60)
    df_treino = splits_norm["train_smote"]
    colunas_features = [c for c in df_treino.columns if c != "fault_type"]
    logger.info("Colunas geradas (%d): %s", len(colunas_features), colunas_features)
    logger.info("Distribuição de classes (treino SMOTE):")
    for classe, n in df_treino["fault_type"].value_counts().items():
        logger.info("  %-8s → %d amostras", classe, n)
    logger.info("=" * 60)
    logger.info("Fase 4B concluída. Arquivos prontos para a Fase 5 (IGA-XGBoost).")


if __name__ == "__main__":
    main()
