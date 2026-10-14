"""
Pré-processamento da base de classificação DGA (Estágio 2 — IGA-XGBoost).

Etapas:
  1. Carga e tradução de rótulos (chinês → português)
  2. Deduplicação (mesma classe: mantém 1ª; classe conflitante: remove todas)
  3. Split estratificado treino/val/teste (70/15/15) por fault_type
  4. SMOTE apenas no treino (balanceamento de classes minoritárias)

IMPORTANTE: Os CSVs de saída contêm gases em PPM BRUTO.
A normalização (log1p + MinMaxScaler) será aplicada na Fase 4, depois do
cálculo das razões IEC 60599 que exigem valores reais (não transformados).
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Garante que importações relativas funcionam ao rodar via -m
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    CLASSIFICATION_FILE,
    CLASSIFICATION_GASES,
    DATA_PROCESSED,
    MAPA_ROTULOS_ZH,
    RANDOM_SEED,
    SPLIT_TESTE,
    SPLIT_VALIDACAO,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Carga e tradução de rótulos
# ---------------------------------------------------------------------------

def carregar_dataset(caminho: Path) -> pd.DataFrame:
    """Lê o Excel e traduz a coluna de rótulos ZH → PT."""
    logger.info("Carregando %s ...", caminho)
    df = pd.read_excel(caminho)

    # Detecta a coluna de rótulo (coluna com valores presentes em MAPA_ROTULOS_ZH)
    col_rotulo = None
    for col in df.columns:
        if df[col].isin(MAPA_ROTULOS_ZH).any():
            col_rotulo = col
            break
    if col_rotulo is None:
        raise ValueError(
            f"Não foi possível identificar a coluna de rótulo no arquivo: {caminho}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df = df.rename(columns={col_rotulo: "fault_type"})
    df["fault_type"] = df["fault_type"].map(MAPA_ROTULOS_ZH)

    nulos = df["fault_type"].isna().sum()
    if nulos > 0:
        raise ValueError(
            f"{nulos} rótulo(s) não mapeado(s) em MAPA_ROTULOS_ZH. "
            f"Verifique os valores únicos da coluna de rótulo."
        )

    faltando = [g for g in CLASSIFICATION_GASES if g not in df.columns]
    if faltando:
        raise ValueError(f"Colunas de gás ausentes no dataset: {faltando}")

    logger.info(
        "Dataset carregado: %d amostras, %d classes únicas",
        len(df), df["fault_type"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# 2. Deduplicação
# ---------------------------------------------------------------------------

def deduplicar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicatas exatas nos 5 gases.

    Estratégia:
    - Mesma classe: mantém a 1ª ocorrência, descarta as demais.
    - Classes conflitantes (mesmo padrão de gás, rótulos diferentes):
      remove TODAS as ocorrências.

    Returns
    -------
    df_limpo : pd.DataFrame
    relatorio : dict
    """
    chave = CLASSIFICATION_GASES

    # Marca como duplicata toda linha que tenha ao menos uma outra igual
    dup_mask = df.duplicated(subset=chave, keep=False)
    df_dup = df[dup_mask].copy()
    n_dup_brutas = len(df_dup)

    idx_remover = set()

    for _, grupo in df_dup.groupby(chave):
        classes_unicas = grupo["fault_type"].unique()
        if len(classes_unicas) == 1:
            # Mesma classe: descarta todas as cópias exceto a 1ª
            idx_remover.update(grupo.index[1:].tolist())
        else:
            # Classes conflitantes: descarta TODAS as ocorrências
            idx_remover.update(grupo.index.tolist())

    n_conflito = sum(
        len(grupo)
        for _, grupo in df_dup.groupby(chave)
        if len(grupo["fault_type"].unique()) > 1
    )

    df_limpo = df.drop(index=list(idx_remover)).reset_index(drop=True)

    relatorio = {
        "total_original": int(len(df)),
        "duplicatas_brutas_identificadas": int(n_dup_brutas),
        "linhas_removidas_mesma_classe": int(len(idx_remover) - n_conflito),
        "linhas_removidas_conflito": int(n_conflito),
        "total_removido": int(len(idx_remover)),
        "total_apos_dedup": int(len(df_limpo)),
        "distribuicao_classes_apos_dedup": {
            k: int(v)
            for k, v in df_limpo["fault_type"].value_counts().items()
        },
    }

    logger.info(
        "Deduplicação: %d removidas (%d mesma-classe + %d conflitantes) → %d restantes",
        relatorio["total_removido"],
        relatorio["linhas_removidas_mesma_classe"],
        relatorio["linhas_removidas_conflito"],
        relatorio["total_apos_dedup"],
    )
    return df_limpo, relatorio


# ---------------------------------------------------------------------------
# 3. Split estratificado treino / val / teste
# ---------------------------------------------------------------------------

def fazer_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado por fault_type: 70 / 15 / 15."""
    # 1º split: separa teste (15%)
    df_trainval, df_teste = train_test_split(
        df,
        test_size=SPLIT_TESTE,
        stratify=df["fault_type"],
        random_state=RANDOM_SEED,
    )
    # 2º split: separa validação (15% do total ≈ 17.6% do trainval)
    val_fraction = SPLIT_VALIDACAO / (1.0 - SPLIT_TESTE)
    df_treino, df_val = train_test_split(
        df_trainval,
        test_size=val_fraction,
        stratify=df_trainval["fault_type"],
        random_state=RANDOM_SEED,
    )
    logger.info(
        "Split: treino=%d | val=%d | teste=%d (total=%d)",
        len(df_treino), len(df_val), len(df_teste),
        len(df_treino) + len(df_val) + len(df_teste),
    )
    return df_treino, df_val, df_teste


# ---------------------------------------------------------------------------
# 4. SMOTE — apenas no treino
# ---------------------------------------------------------------------------

def aplicar_smote(df_treino: pd.DataFrame) -> pd.DataFrame:
    """Aplica SMOTE no treino para balancear classes minoritárias.

    SMOTE nunca é aplicado em val/teste.
    """
    X = df_treino[CLASSIFICATION_GASES].values
    y = df_treino["fault_type"].values

    smote = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X, y)

    df_smote = pd.DataFrame(X_res, columns=CLASSIFICATION_GASES)
    df_smote["fault_type"] = y_res

    logger.info(
        "SMOTE: %d → %d amostras | distribuição pós-SMOTE: %s",
        len(df_treino),
        len(df_smote),
        df_smote["fault_type"].value_counts().to_dict(),
    )
    return df_smote


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def executar(salvar_relatorio: bool = True) -> None:
    """Executa o pipeline completo de pré-processamento da base de classificação."""
    df = carregar_dataset(CLASSIFICATION_FILE)
    df_limpo, relatorio_dedup = deduplicar(df)
    df_treino, df_val, df_teste = fazer_split(df_limpo)
    df_treino_smote = aplicar_smote(df_treino)

    # Salvar CSVs em PPM BRUTO (normalização fica para a Fase 4)
    colunas_saida = CLASSIFICATION_GASES + ["fault_type"]
    saidas = {
        "classification_train.csv": df_treino[colunas_saida],
        "classification_train_smote.csv": df_treino_smote[colunas_saida],
        "classification_val.csv": df_val[colunas_saida],
        "classification_test.csv": df_teste[colunas_saida],
    }
    for nome, dados in saidas.items():
        destino = DATA_PROCESSED / nome
        dados.to_csv(destino, index=False)
        logger.info("Salvo: %s (%d linhas)", destino, len(dados))

    if salvar_relatorio:
        relatorio_path = DATA_PROCESSED / "dedup_report.json"
        with open(relatorio_path, "w", encoding="utf-8") as f:
            json.dump(relatorio_dedup, f, ensure_ascii=False, indent=2)
        logger.info("Relatório de deduplicação salvo em %s", relatorio_path)

    logger.info("Pré-processamento da base de classificação concluído.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    executar()
