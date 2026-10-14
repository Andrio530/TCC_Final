"""
Ponto de entrada da Fase 4 — Engenharia de Atributos.

Execute este script no Colab (onde o TensorFlow está disponível) ou localmente
se o TensorFlow estiver instalado. Gera os CSVs de features prontos para
a Fase 5 (IGA-XGBoost).

Uso:
  python -m src.feature_engineering.run_feature_engineering
  # ou no Colab:
  !python -m src.feature_engineering.run_feature_engineering
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Fase 4 - Engenharia de Atributos")
    logger.info("=" * 60)

    # Track A: features para o sistema hibrido (base Katser)
    # Requer TensorFlow instalado (rodar no Colab)
    logger.info("Iniciando Track A (Katser - sistema hibrido)...")
    from src.feature_engineering.build_katser_features import main as run_katser  # noqa: PLC0415
    run_katser()

    logger.info("=" * 60)

    # Track B: features para o classificador ampliado (base chinesa)
    # Nao requer TensorFlow - pode rodar localmente
    logger.info("Iniciando Track B (China - classificador ampliado)...")
    from src.feature_engineering.build_classification_features import main as run_china  # noqa: PLC0415
    run_china()

    logger.info("=" * 60)
    logger.info("Fase 4 concluida. Arquivos gerados em data/processed/:")
    logger.info("  [Track A] katser_features_{train,val,test}.csv")
    logger.info("  [Track A] katser_feature_scaler.pkl")
    logger.info("  [Track B] china_features_{train,train_smote,val,test}.csv")
    logger.info("  [Track B] china_feature_scaler.pkl")


if __name__ == "__main__":
    main()
