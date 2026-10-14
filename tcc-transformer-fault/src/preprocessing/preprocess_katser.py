"""
Pré-processamento da base Katser de séries temporais (Estágio 1 — LSTM).

Etapas:
  1. Carga: 3.000 CSVs (2.100 treino + 900 teste) + labels FDD
  2. Split aleatório estratificado por category dentro das 2.100 de treino
     (15% → validação, ~315 séries)
  3. Persistência em arrays NumPy comprimidos (.npz)

Notas sobre o split:
  - A estratificação é por category (1=Normal, 2=PD, 3=D1, 4=T1).
  - Split ALEATÓRIO — a ordem de ID NÃO é usada como proxy cronológico entre
    séries, pois cada série representa um equipamento diferente e não há
    evidência de que os IDs sejam cronológicos entre equipamentos distintos.
  - A cronologia walk-forward aplica-se DENTRO de cada série (420 timesteps),
    não entre séries, e será respeitada pelo gerador de janelas (windowing.py).
  - As 900 séries de teste oficial permanecem intocadas.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_PROCESSED,
    KATSER_DIR,
    KATSER_GASES,
    KATSER_CLASSES,
    RANDOM_SEED,
    SPLIT_VAL_KATSER,
)

logger = logging.getLogger(__name__)

# Colunas esperadas nos CSVs da base Katser (4 gases, sem coluna de tempo própria)
_KATSER_COL_MAP = {
    "H2": "H2",
    "CO": "CO",
    "C2H4": "C2H4",
    "C2H2": "C2H2",
}
N_TIMESTEPS = 420  # número de pontos por série


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------

def _carregar_labels(split: str) -> pd.DataFrame:
    """Carrega o arquivo de labels FDD para o split indicado ('train' ou 'test')."""
    caminho = KATSER_DIR / f"labels_fdd_{split}.csv"
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de labels não encontrado: {caminho}")
    df = pd.read_csv(caminho)
    return df


def _detectar_colunas_gases(df_csv: pd.DataFrame) -> list[str]:
    """Detecta as colunas de gás no CSV (tolerante a variações de nome)."""
    cols = []
    for gas in KATSER_GASES:
        candidatas = [c for c in df_csv.columns if gas.lower() in c.lower()]
        if not candidatas:
            raise ValueError(
                f"Coluna para gás '{gas}' não encontrada. "
                f"Colunas disponíveis: {list(df_csv.columns)}"
            )
        cols.append(candidatas[0])
    return cols


def _carregar_serie(caminho_csv: Path) -> np.ndarray:
    """Lê um CSV de série e retorna array (N_TIMESTEPS, n_gases).

    Raises
    ------
    ValueError
        Se a série não tiver exatamente N_TIMESTEPS linhas.
    """
    df = pd.read_csv(caminho_csv)
    colunas = _detectar_colunas_gases(df)
    dados = df[colunas].values.astype(np.float32)

    if dados.shape[0] != N_TIMESTEPS:
        raise ValueError(
            f"{caminho_csv.name}: esperado {N_TIMESTEPS} timesteps, "
            f"encontrado {dados.shape[0]}."
        )
    return dados


def carregar_base_katser(split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Carrega todas as séries de um split ('train' ou 'test').

    Returns
    -------
    series : np.ndarray
        Array de shape (n_series, N_TIMESTEPS, n_gases).
    labels : np.ndarray
        Array int de shape (n_series,) com a categoria FDD de cada série.
    df_labels : pd.DataFrame
        DataFrame original de labels (para uso no relatório de split).
    """
    pasta = KATSER_DIR / f"data_{split}"
    if not pasta.exists():
        raise FileNotFoundError(f"Diretório de dados não encontrado: {pasta}")

    df_labels = _carregar_labels(split)

    # Coluna de nome de arquivo e de category
    col_file = next((c for c in df_labels.columns if "file" in c.lower() or c.lower() == "id"), None)
    col_cat  = next((c for c in df_labels.columns if "categ" in c.lower()), None)
    if col_file is None or col_cat is None:
        raise ValueError(
            f"Não foi possível identificar colunas de arquivo/categoria em "
            f"labels_fdd_{split}.csv. Colunas: {list(df_labels.columns)}"
        )

    series_list = []
    labels_list = []
    erros = []

    for _, row in df_labels.iterrows():
        nome_arquivo = row[col_file]
        categoria = int(row[col_cat])
        caminho_csv = pasta / nome_arquivo

        if not caminho_csv.exists():
            erros.append(nome_arquivo)
            continue

        try:
            serie = _carregar_serie(caminho_csv)
        except ValueError as e:
            logger.warning("Pulando %s: %s", nome_arquivo, e)
            continue

        series_list.append(serie)
        labels_list.append(categoria)

    if erros:
        logger.warning("%d arquivo(s) não encontrado(s): %s", len(erros), erros[:5])

    series_arr = np.stack(series_list, axis=0)   # (n, 420, 4)
    labels_arr = np.array(labels_list, dtype=np.int32)  # (n,)

    logger.info(
        "Katser %s: %d séries carregadas | shape=%s | classes=%s",
        split,
        len(series_arr),
        series_arr.shape,
        {k: int((labels_arr == k).sum()) for k in np.unique(labels_arr)},
    )
    return series_arr, labels_arr, df_labels


# ---------------------------------------------------------------------------
# 2. Split aleatório estratificado
# ---------------------------------------------------------------------------

def fazer_split_katser(
    series: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide as séries de treino em treino/validação de forma aleatória estratificada.

    O split é estratificado por category (1=Normal, 2=PD, 3=D1, 4=T1).
    A ordem de ID não é usada — cada série representa um equipamento
    independente e não há evidência de sequência cronológica entre elas.

    Returns
    -------
    s_treino, l_treino, s_val, l_val : np.ndarray
    """
    indices = np.arange(len(series))

    idx_treino, idx_val = train_test_split(
        indices,
        test_size=SPLIT_VAL_KATSER,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    s_treino = series[idx_treino]
    l_treino = labels[idx_treino]
    s_val    = series[idx_val]
    l_val    = labels[idx_val]

    logger.info(
        "Split Katser: treino=%d | val=%d",
        len(s_treino), len(s_val),
    )
    return s_treino, l_treino, s_val, l_val


# ---------------------------------------------------------------------------
# 3. Persistência
# ---------------------------------------------------------------------------

def salvar_npz(
    series: np.ndarray,
    labels: np.ndarray,
    nome: str,
) -> Path:
    """Salva séries e labels em arquivo .npz comprimido."""
    destino = DATA_PROCESSED / nome
    np.savez_compressed(destino, series=series, labels=labels)
    logger.info("Salvo: %s | shape=%s", destino, series.shape)
    return destino


def _distribuicao_classes(labels: np.ndarray) -> dict:
    """Retorna distribuição de classes como dict legível."""
    dist = {}
    for k in np.unique(labels):
        nome = KATSER_CLASSES.get(int(k), f"classe_{k}")
        dist[nome] = int((labels == k).sum())
    return dist


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def executar() -> None:
    """Executa o pipeline completo de pré-processamento da base Katser."""
    # Treino (2100 séries originais)
    s_train_all, l_train_all, _ = carregar_base_katser("train")

    # Split aleatório estratificado
    s_treino, l_treino, s_val, l_val = fazer_split_katser(s_train_all, l_train_all)

    # Teste oficial (intocado)
    s_teste, l_teste, _ = carregar_base_katser("test")

    # Persistência
    salvar_npz(s_treino, l_treino, "katser_train.npz")
    salvar_npz(s_val,    l_val,    "katser_val.npz")
    salvar_npz(s_teste,  l_teste,  "katser_test.npz")

    # Relatório de split
    relatorio = {
        "total_series": int(len(s_train_all) + len(s_teste)),
        "treino": {
            "n_series": int(len(s_treino)),
            "shape": list(s_treino.shape),
            "distribuicao_classes": _distribuicao_classes(l_treino),
        },
        "validacao": {
            "n_series": int(len(s_val)),
            "shape": list(s_val.shape),
            "distribuicao_classes": _distribuicao_classes(l_val),
        },
        "teste": {
            "n_series": int(len(s_teste)),
            "shape": list(s_teste.shape),
            "distribuicao_classes": _distribuicao_classes(l_teste),
            "nota": "Conjunto de teste oficial do Katser — intocado",
        },
        "metadados": {
            "split_val_katser": SPLIT_VAL_KATSER,
            "random_seed": RANDOM_SEED,
            "estrategia_split": "aleatório estratificado por category",
            "n_timesteps_por_serie": N_TIMESTEPS,
            "gases": KATSER_GASES,
        },
    }

    relatorio_path = DATA_PROCESSED / "katser_split_report.json"
    with open(relatorio_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    logger.info("Relatório de split salvo em %s", relatorio_path)

    logger.info("Pré-processamento da base Katser concluído.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    executar()
