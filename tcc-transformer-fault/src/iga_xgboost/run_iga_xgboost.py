"""
Ponto de entrada da Fase 5 — IGA-XGBoost.

Executa sequencialmente:
  Track A: sistema híbrido completo (LSTM → IGA-XGBoost) na base Katser
  Track B: classificador ampliado (IGA-XGBoost) na base chinesa

Uso:
  python -m src.iga_xgboost.run_iga_xgboost

  # Para executar apenas um track:
  python -m src.iga_xgboost.run_iga_xgboost --track A
  python -m src.iga_xgboost.run_iga_xgboost --track B
"""

import argparse
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
    parser = argparse.ArgumentParser(
        description="Fase 5 — IGA-XGBoost (Track A e/ou Track B)"
    )
    parser.add_argument(
        "--track",
        choices=["A", "B", "ambos"],
        default="ambos",
        help="Qual track executar (padrão: ambos).",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Fase 5 — IGA-XGBoost")
    logger.info("=" * 60)

    if args.track in ("A", "ambos"):
        logger.info("Iniciando Track A (Katser — sistema híbrido)...")
        from src.iga_xgboost.train_hybrid import main as run_track_a  # noqa: PLC0415
        resultados_a = run_track_a()
        logger.info(
            "Track A concluído — IGA F1=%.4f | Default F1=%.4f",
            resultados_a["C1_IGA"]["multiseed"]["f1_macro_mean"],
            resultados_a["C2_default"]["multiseed"]["f1_macro_mean"],
        )

    logger.info("=" * 60)

    if args.track in ("B", "ambos"):
        logger.info("Iniciando Track B (China — classificador ampliado)...")
        from src.iga_xgboost.train_classification import main as run_track_b  # noqa: PLC0415
        resultados_b = run_track_b()

        # Resumo comparativo
        c3 = resultados_b["C3_IGA"]["multiseed"]["f1_macro_mean"]
        c4 = resultados_b["C4_default"]["multiseed"]["f1_macro_mean"]
        iec = resultados_b.get("T1_IEC60599", {}).get("somente_falhas", {}).get("f1_macro", float("nan"))
        duval = resultados_b.get("T2_Duval", {}).get("somente_falhas", {}).get("f1_macro", float("nan"))
        logger.info(
            "Track B concluído — IGA F1=%.4f | Default F1=%.4f | "
            "IEC60599 F1(falhas)=%.4f | Duval F1(falhas)=%.4f",
            c3, c4, iec, duval,
        )

    logger.info("=" * 60)
    logger.info("Fase 5 concluída. Resultados em reports/metrics/:")
    if args.track in ("A", "ambos"):
        logger.info("  iga_track_a_results.json")
        logger.info("  iga_evolution_track_a.json")
    if args.track in ("B", "ambos"):
        logger.info("  iga_track_b_results.json")
        logger.info("  iga_evolution_track_b.json")


if __name__ == "__main__":
    main()
