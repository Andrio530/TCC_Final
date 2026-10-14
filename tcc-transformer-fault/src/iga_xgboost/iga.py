"""
Motor do IGA (Improved Genetic Algorithm).

Otimiza simultaneamente a seleção de features e os hiperparâmetros do XGBoost
usando um algoritmo genético com os seguintes operadores:

  Seleção  : Torneio (k = 3)
  Crossover: Uniforme (parte binária) + BLX-α com α = 0.5 (parte contínua)
  Mutação  : Bit-flip por posição (parte binária) + Gaussiana (parte contínua)
  Elitismo : Os 2 melhores indivíduos passam diretamente para a próxima geração
  Fitness  : F1-macro via StratifiedKFold(5) no conjunto de treino

O que torna este GA "improved":
  1. Inicialização enviesada por conhecimento de domínio (razões IEC com p = 0.8).
  2. Elitismo garantido.
  3. Early stopping por estagnação de fitness.
"""

import logging
import time
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    IGA_GERACOES_SEM_MELHORIA,
    IGA_MAX_GERACOES,
    IGA_POPULACAO,
    IGA_PROB_ATIVACAO_DEMAIS,
    IGA_PROB_ATIVACAO_RAZOES_IEC,
    RANDOM_SEED,
)
from src.iga_xgboost.cromossomo import (  # noqa: E402
    _specs_do_track,
    cromossomo_len,
    decodificar,
    gerar_cromossomo,
    validar,
)

logger = logging.getLogger(__name__)

# Número de folds para avaliação do fitness
# Número de folds para avaliação do fitness
N_FOLDS_FITNESS = 5

# Probabilidade de crossover por par de pais
P_CROSSOVER = 0.9

# Probabilidade de mutação por bit (parte binária)
P_MUTACAO_BIT = 0.1

# Desvio-padrão do ruído gaussiano na mutação (parte contínua)
SIGMA_MUTACAO = 0.1

# α para BLX-α crossover (contínuo)
BLX_ALPHA = 0.5

# Número de indivíduos de elite preservados por elitismo
N_ELITE = 2

# Tamanho do torneio de seleção
K_TORNEIO = 3


class IGA:
    """Improved Genetic Algorithm para seleção de features e otimização de hiperparâmetros.

    Parameters
    ----------
    track : str
        'A' (Katser, 4 classes) ou 'B' (China, 7 classes).
    X_train : np.ndarray
        Features de treino, shape (n_amostras, n_features_disponíveis).
    y_train : np.ndarray
        Labels de treino como inteiros, shape (n_amostras,).
    seed : int
        Seed para reprodutibilidade completa.
    """

    def __init__(
        self,
        track: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int = RANDOM_SEED,
        usar_smote: bool = True,
        smote_k_neighbors: int = 5,
        sample_weight: np.ndarray | None = None,
        smote_sampler = None,
    ) -> None:
        self.track = track
        self.X_train = X_train
        self.y_train = y_train
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.usar_smote = usar_smote
        self.smote_k_neighbors = smote_k_neighbors
        self.sample_weight = sample_weight
        self.smote_sampler = smote_sampler

        self.features, _ = _specs_do_track(track)
        self.n_features = len(self.features)
        self.n_cromossomo = cromossomo_len(track)

        # Estado do algoritmo
        self.historico_melhor: list[float] = []   # F1 do melhor por geração
        self.historico_media: list[float] = []    # F1 médio por geração
        self.melhor_cromossomo: np.ndarray | None = None
        self.melhor_fitness: float = -np.inf

    # -----------------------------------------------------------------------
    # Fitness
    # -----------------------------------------------------------------------

    def _avaliar(self, cromossomo: np.ndarray) -> float:
        """Calcula o fitness (F1-macro) de um cromossomo via cross-validation.

        Retorna 0.0 para cromossomos inviáveis (< MIN_FEATURES_ATIVAS).
        """
        if not validar(cromossomo, self.track):
            return 0.0

        features_ativas, hparams = decodificar(cromossomo, self.track)
        idx_features = [self.features.index(f) for f in features_ativas]
        
        # Seleção de features move-se para dentro do pipeline
        # Assim o SMOTE (especialmente CustomSMOTE) tem acesso a todas as features
        selector = FunctionTransformer(lambda X, idx=idx_features: X[:, idx])

        clf = XGBClassifier(
            **hparams,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=self.seed,
            n_jobs=1,
            verbosity=0,
        )

        if self.usar_smote:
            sampler = self.smote_sampler if self.smote_sampler is not None else SMOTE(random_state=self.seed, k_neighbors=self.smote_k_neighbors)
            pipeline = ImbPipeline([
                ("smote", sampler),
                ("select", selector),
                ("clf", clf),
            ])
            fit_params = {}
        else:
            pipeline = ImbPipeline([
                ("select", selector),
                ("clf", clf),
            ])
            fit_params = {"clf__sample_weight": self.sample_weight} if self.sample_weight is not None else {}

        cv = StratifiedKFold(n_splits=N_FOLDS_FITNESS, shuffle=True, random_state=self.seed)
        
        import sklearn
        from packaging import version
        
        cv_kwargs = {}
        if version.parse(sklearn.__version__) >= version.parse("1.4"):
            cv_kwargs["params"] = fit_params
        else:
            cv_kwargs["fit_params"] = fit_params
            
        try:
            scores = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=cv, scoring="f1_macro", n_jobs=-1,  # folds em paralelo
                **cv_kwargs
            )
            return float(np.clip(scores.mean(), 0.0, 1.0))
        except Exception as exc:
            logger.error("Erro ao avaliar cromossomo: %s", exc)
            return 0.0

    def _avaliar_populacao(self, populacao: list[np.ndarray]) -> list[float]:
        return [self._avaliar(c) for c in populacao]

    # -----------------------------------------------------------------------
    # Inicialização
    # -----------------------------------------------------------------------

    def _inicializar(self) -> list[np.ndarray]:
        return [
            gerar_cromossomo(
                self.track,
                self.rng,
                prob_iec=IGA_PROB_ATIVACAO_RAZOES_IEC,
                prob_demais=IGA_PROB_ATIVACAO_DEMAIS,
            )
            for _ in range(IGA_POPULACAO)
        ]

    # -----------------------------------------------------------------------
    # Seleção
    # -----------------------------------------------------------------------

    def _selecionar(
        self,
        populacao: list[np.ndarray],
        fitnesses: list[float],
    ) -> np.ndarray:
        """Seleção por torneio (k = K_TORNEIO)."""
        candidatos = self.rng.choice(len(populacao), size=K_TORNEIO, replace=False)
        vencedor = max(candidatos, key=lambda i: fitnesses[i])
        return populacao[vencedor].copy()

    # -----------------------------------------------------------------------
    # Crossover
    # -----------------------------------------------------------------------

    def _cruzar(
        self,
        pai1: np.ndarray,
        pai2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crossover uniforme (binário) + BLX-α (contínuo).

        Se sorteio > P_CROSSOVER, devolve cópias dos pais sem cruzamento.
        """
        if self.rng.random() > P_CROSSOVER:
            return pai1.copy(), pai2.copy()

        f1 = pai1.copy()
        f2 = pai2.copy()

        # Parte binária: crossover uniforme (troca bit a bit com p=0.5)
        for i in range(self.n_features):
            if self.rng.random() < 0.5:
                f1[i], f2[i] = f2[i], f1[i]

        # Parte contínua: BLX-α
        for i in range(self.n_features, self.n_cromossomo):
            lo = min(pai1[i], pai2[i])
            hi = max(pai1[i], pai2[i])
            ext = BLX_ALPHA * (hi - lo)
            f1[i] = np.clip(self.rng.uniform(lo - ext, hi + ext), 0.0, 1.0)
            f2[i] = np.clip(self.rng.uniform(lo - ext, hi + ext), 0.0, 1.0)

        return f1, f2

    # -----------------------------------------------------------------------
    # Mutação
    # -----------------------------------------------------------------------

    def _mutar(self, cromossomo: np.ndarray) -> np.ndarray:
        """Mutação bit-flip (binário) + Gaussiana (contínuo)."""
        c = cromossomo.copy()

        # Parte binária: bit-flip com P_MUTACAO_BIT por posição
        for i in range(self.n_features):
            if self.rng.random() < P_MUTACAO_BIT:
                c[i] = 1.0 - c[i]

        # Parte contínua: ruído gaussiano com clipping
        for i in range(self.n_features, self.n_cromossomo):
            c[i] = np.clip(c[i] + self.rng.normal(0.0, SIGMA_MUTACAO), 0.0, 1.0)

        return c

    # -----------------------------------------------------------------------
    # Loop evolutivo principal
    # -----------------------------------------------------------------------

    def evoluir(self) -> np.ndarray:
        """Executa o loop evolutivo e retorna o melhor cromossomo encontrado.

        Returns
        -------
        melhor_cromossomo : np.ndarray
            Cromossomo com maior F1-macro no conjunto de treino (CV).
        """
        logger.info(
            "IGA iniciado — track=%s | pop=%d | max_ger=%d | early_stop=%d",
            self.track, IGA_POPULACAO, IGA_MAX_GERACOES, IGA_GERACOES_SEM_MELHORIA,
        )

        populacao = self._inicializar()
        fitnesses = self._avaliar_populacao(populacao)

        geracoes_sem_melhoria = 0
        t0 = time.time()

        for geracao in range(IGA_MAX_GERACOES):
            # Ordena por fitness (decrescente)
            ranking = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)

            # Atualiza melhor global
            if fitnesses[ranking[0]] > self.melhor_fitness:
                self.melhor_fitness = fitnesses[ranking[0]]
                self.melhor_cromossomo = populacao[ranking[0]].copy()
                geracoes_sem_melhoria = 0
            else:
                geracoes_sem_melhoria += 1

            self.historico_melhor.append(self.melhor_fitness)
            self.historico_media.append(float(np.mean(fitnesses)))

            logger.info(
                "Geração %3d | F1 melhor=%.4f | F1 médio=%.4f | sem melhoria=%d | %.0fs",
                geracao + 1,
                self.melhor_fitness,
                self.historico_media[-1],
                geracoes_sem_melhoria,
                time.time() - t0,
            )

            if geracoes_sem_melhoria >= IGA_GERACOES_SEM_MELHORIA:
                logger.info("Early stopping ativado na geração %d.", geracao + 1)
                break

            # Nova geração com elitismo
            nova_pop = [populacao[i].copy() for i in ranking[:N_ELITE]]

            while len(nova_pop) < IGA_POPULACAO:
                pai1 = self._selecionar(populacao, fitnesses)
                pai2 = self._selecionar(populacao, fitnesses)
                filho1, filho2 = self._cruzar(pai1, pai2)
                nova_pop.append(self._mutar(filho1))
                if len(nova_pop) < IGA_POPULACAO:
                    nova_pop.append(self._mutar(filho2))

            populacao = nova_pop
            fitnesses = self._avaliar_populacao(populacao)

        features_selecionadas, hparams = decodificar(self.melhor_cromossomo, self.track)
        logger.info(
            "IGA concluído — F1 CV=%.4f | features (%d): %s | params: %s",
            self.melhor_fitness,
            len(features_selecionadas),
            features_selecionadas,
            hparams,
        )

        return self.melhor_cromossomo
