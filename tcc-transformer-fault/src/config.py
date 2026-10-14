"""
Configuração central do projeto. Todo script/módulo que precise de seed,
caminhos ou constantes do domínio (gases, classes de falha) deve importar
daqui, para evitar duplicação e inconsistência entre módulos.
"""
from pathlib import Path

RANDOM_SEED = 42

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_EXTERNAL = ROOT_DIR / "data" / "external"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Constantes de gases
# ---------------------------------------------------------------------------

# REFERÊNCIA TEÓRICA DO TCC I — NÃO usar em código funcional.
# Os 7 gases canônicos da norma IEC 60599 foram citados no TCC I (Seção 3.5.1),
# mas nenhuma das bases reais disponíveis contém todos os 7 simultaneamente.
# Use CLASSIFICATION_GASES ou KATSER_GASES conforme a base de dados.
GASES_REFERENCIA_TCC_I = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]

# Gases disponíveis na base de classificação (Estágio 2 — IGA-XGBoost)
# Fonte: dataset_2321_7classes.xlsx (alan-456/transformer-fault-dataset)
CLASSIFICATION_GASES = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]

# Gases disponíveis na base Katser (Estágio 1 — LSTM)
# Fonte: Katser et al. 2022, arXiv 2211.11013
KATSER_GASES = ["H2", "CO", "C2H4", "C2H2"]

# ---------------------------------------------------------------------------
# Razões de gases — base de classificação (Track B)
# ---------------------------------------------------------------------------
# Fontes: IEC 60599, Rogers (1975), Doernenburg (IEEE), Wu et al. (2021).
# CO2/CO foi removida: nenhuma das bases tem CO e CO2 simultaneamente.
# A base de classificação tem CO2 mas não CO; a base Katser tem CO mas não CO2.
RAZOES_CLASSIFICACAO = [
    "CH4/H2",       # IEC 60599, Rogers
    "C2H2/C2H4",    # IEC 60599, Rogers
    "C2H4/C2H6",    # IEC 60599, Rogers
    "C2H2/CH4",     # Doernenburg
    "C2H2/C2H6",    # Doernenburg / Wu et al. (2021) — inverso do Doernenburg original
]

# Features de soma de hidrocarbonetos e percentuais relativos (Wu et al. 2021, Tabela 1)
# TH  = CH4 + C2H4 + C2H2        (gases do Triângulo de Duval)
# TH1 = CH4 + C2H4 + C2H2 + C2H6 (todos os hidrocarbonetos combustíveis)
TH_FEATURES = [
    "TH", "TH1",
    "CH4/TH", "C2H4/TH", "C2H2/TH",
    "CH4/TH1", "C2H6/TH1", "C2H4/TH1", "C2H2/TH1",
]

# ---------------------------------------------------------------------------
# Classes de falha
# ---------------------------------------------------------------------------
CLASSES_FALHA = ["Normal", "PD", "D1", "D2", "T1", "T2", "T3"]

# Mapeamento dos rótulos originais em chinês (dataset compilado, ver data/README.md)
MAPA_ROTULOS_ZH = {
    "正常": "Normal",
    "局部放电": "PD",   # descarga parcial
    "低能放电": "D1",   # descarga de baixa energia
    "高能放电": "D2",   # descarga de alta energia
    "低温过热": "T1",   # sobreaquecimento baixa temperatura (<300C)
    "中温过热": "T2",   # sobreaquecimento média temperatura (300-700C)
    "高温过热": "T3",   # sobreaquecimento alta temperatura (>700C)
}

# Mapeamento das categorias numéricas da base Katser
# Ref: labels_fdd_train.csv / labels_fdd_test.csv
KATSER_CLASSES = {1: "Normal", 2: "PD", 3: "D1", 4: "T1"}

# ---------------------------------------------------------------------------
# Caminhos das bases de dados
# ---------------------------------------------------------------------------
KATSER_DIR = DATA_EXTERNAL / "katser_npp_transformers"
CLASSIFICATION_FILE = DATA_EXTERNAL / "dga_compiled_zh" / "dataset_2321_7classes.xlsx"

# ---------------------------------------------------------------------------
# Parâmetros de pré-processamento e split
# ---------------------------------------------------------------------------
MAX_GAP_INTERPOLACAO = 3

SPLIT_TREINO = 0.70
SPLIT_VALIDACAO = 0.15
SPLIT_TESTE = 0.15

# Fração de validação dentro das 2.100 séries de treino do Katser
# Split aleatório estratificado por category — não por ordem de ID.
SPLIT_VAL_KATSER = 0.15

# Tamanhos de janela deslizante a testar na Fase 3 (avaliados por MAE/RMSE da LSTM)
JANELAS_CANDIDATAS = [10, 20, 30, 50]

# ---------------------------------------------------------------------------
# Parâmetros da LSTM (Fase 3)
# ---------------------------------------------------------------------------
LSTM_UNITS = 64            # Unidades por camada LSTM
LSTM_CAMADAS = 2           # Número de camadas empilhadas (Seção 3.5 do TCC)
LSTM_DROPOUT = 0.2         # Dropout entre camadas (regularização)
LSTM_HORIZONTE = 1         # Passos à frente (h=1 → próximo timestep = 12h)
LSTM_EPOCHS_MAX = 200      # Máximo de épocas (early stopping cortará antes)
LSTM_BATCH_SIZE = 1024       # Batch size
LSTM_LEARNING_RATE = 0.001 # Learning rate do Adam
LSTM_PATIENCE = 15         # Early stopping: épocas sem melhoria no val_loss
LSTM_MIN_DELTA = 1e-6      # Melhoria mínima para resetar o patience

# Seeds para avaliação com múltiplas execuções (item 2 do validation_checklist.md)
SEEDS_AVALIACAO = [42, 123, 456, 789, 2024]

# ---------------------------------------------------------------------------
# Parâmetros do IGA (Fase 5)
# ---------------------------------------------------------------------------
IGA_POPULACAO = 50
IGA_MAX_GERACOES = 100
IGA_GERACOES_SEM_MELHORIA = 20
IGA_PROB_ATIVACAO_RAZOES_IEC = 0.8
IGA_PROB_ATIVACAO_DEMAIS = 0.5

for _dir in (DATA_PROCESSED, METRICS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
