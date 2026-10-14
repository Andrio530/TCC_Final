"""
Orquestrador do pipeline de pré-processamento (Fase 2).

Uso:
    # Executar tudo
    python -m src.preprocessing.run_preprocessing

    # Executar apenas a base de classificação
    python -m src.preprocessing.run_preprocessing --classification

    # Executar apenas a base Katser
    python -m src.preprocessing.run_preprocessing --katser
"""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_classification, preprocess_katser


def _configurar_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _executar_etapa(nome: str, fn) -> float:
    """Executa uma etapa do pipeline e retorna o tempo gasto em segundos."""
    logger = logging.getLogger("run_preprocessing")
    logger.info("=" * 60)
    logger.info("Iniciando: %s", nome)
    logger.info("=" * 60)
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    logger.info("Concluído: %s em %.1fs", nome, elapsed)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de pré-processamento — Fase 2 do TCC II."
    )
    parser.add_argument(
        "--classification",
        action="store_true",
        help="Executar apenas o pré-processamento da base de classificação.",
    )
    parser.add_argument(
        "--katser",
        action="store_true",
        help="Executar apenas o pré-processamento da base Katser.",
    )
    args = parser.parse_args()

    _configurar_logging()
    logger = logging.getLogger("run_preprocessing")

    # Se nenhuma flag foi passada, executa tudo
    executar_classification = args.classification or (not args.classification and not args.katser)
    executar_katser          = args.katser          or (not args.classification and not args.katser)

    tempos = {}
    t_total = time.perf_counter()

    if executar_classification:
        tempos["classification"] = _executar_etapa(
            "Base de Classificação (DGA)",
            preprocess_classification.executar,
        )

    if executar_katser:
        tempos["katser"] = _executar_etapa(
            "Base Katser (Séries Temporais)",
            preprocess_katser.executar,
        )

    total = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info("Pipeline concluído em %.1fs", total)
    for etapa, t in tempos.items():
        logger.info("  %-20s %.1fs", etapa, t)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
