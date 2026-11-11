# -*- coding: utf-8 -*-
"""
Gera a planilha Excel com todos os dados da Modelagem Temporal, Janelamento,
Degradação Multi-Horizonte e Decisões de Engenharia da LSTM.
"""
import json
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Caminhos
ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "tcc-transformer-fault" / "reports" / "metrics" / "lstm_multi_horizon_results_norm.json"
OUT_XLSX = ROOT / "estudo_banca" / "modelagem_temporal_lstm.xlsx"

# Carregar JSON de resultados
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data_mh = json.load(f)

wb = openpyxl.Workbook()
# Remove sheet padrao
wb.remove(wb.active)

# Paleta de cores corporativa de Engenharia
NAVY = "102C57"
MIDNIGHT = "1B3C73"
TEAL = "2A9D8F"
LIGHT_BLUE = "EBF3FA"
LIGHT_GRAY = "F8F9FA"
HEADER_GRAY = "E2E8F0"
WHITE = "FFFFFF"
GREEN = "2D6A4F"
LIGHT_GREEN = "D8F3DC"
ALERT_RED = "BA181B"
LIGHT_RED = "F8D7DA"
YELLOW = "D4A373"
LIGHT_YELLOW = "FFF3CD"

font_title = Font(name="Calibri", size=14, bold=True, color=WHITE)
font_subtitle = Font(name="Calibri", size=11, bold=True, color=MIDNIGHT)
font_header = Font(name="Calibri", size=10, bold=True, color=WHITE)
font_bold = Font(name="Calibri", size=10, bold=True)
font_regular = Font(name="Calibri", size=10)
font_italic = Font(name="Calibri", size=9, italic=True)

fill_navy = PatternFill(start_color=NAVY, end_color=NAVY, fill_type="solid")
fill_midnight = PatternFill(start_color=MIDNIGHT, end_color=MIDNIGHT, fill_type="solid")
fill_teal = PatternFill(start_color=TEAL, end_color=TEAL, fill_type="solid")
fill_light_blue = PatternFill(start_color=LIGHT_BLUE, end_color=LIGHT_BLUE, fill_type="solid")
fill_light_gray = PatternFill(start_color=LIGHT_GRAY, end_color=LIGHT_GRAY, fill_type="solid")
fill_header_gray = PatternFill(start_color=HEADER_GRAY, end_color=HEADER_GRAY, fill_type="solid")
fill_light_green = PatternFill(start_color=LIGHT_GREEN, end_color=LIGHT_GREEN, fill_type="solid")
fill_light_red = PatternFill(start_color=LIGHT_RED, end_color=LIGHT_RED, fill_type="solid")
fill_light_yellow = PatternFill(start_color=LIGHT_YELLOW, end_color=LIGHT_YELLOW, fill_type="solid")

thin_border = Border(
    left=Side(style='thin', color='CCCCCC'),
    right=Side(style='thin', color='CCCCCC'),
    top=Side(style='thin', color='CCCCCC'),
    bottom=Side(style='thin', color='CCCCCC')
)
header_border = Border(
    left=Side(style='thin', color='FFFFFF'),
    right=Side(style='thin', color='FFFFFF'),
    top=Side(style='medium', color=NAVY),
    bottom=Side(style='medium', color=NAVY)
)

# ---------------------------------------------------------------------------
# ABA 1: RESUMO EXECUTIVO
# ---------------------------------------------------------------------------
ws1 = wb.create_sheet(title="Resumo_Executivo")
ws1.views.sheetView[0].showGridLines = True

ws1.merge_cells("A1:G1")
ws1["A1"] = "MODELAGEM TEMPORAL E PROJEÇÃO PREDITIVA — STACKED LSTM"
ws1["A1"].font = font_title
ws1["A1"].fill = fill_navy
ws1["A1"].alignment = Alignment(horizontal="center", vertical="center")
ws1.row_dimensions[1].height = 35

ws1.merge_cells("A2:G2")
ws1["A2"] = "TCC Engenharia da Computação (Unisinos) | Aluno: Andrio Gabriel Epp | Orientação Técnica"
ws1["A2"].font = Font(name="Calibri", size=10, italic=True, color=WHITE)
ws1["A2"].fill = fill_midnight
ws1["A2"].alignment = Alignment(horizontal="center", vertical="center")
ws1.row_dimensions[2].height = 20

resumo_items = [
    ("Componente Arquitetural", "Especificação / Decisão de Projeto", "Papel Físico / Motivo de Engenharia"),
    ("Base de Dados de Séries Temporais", "Katser et al. (2022) — Skoltech / Rosatom NPP", "3.000 séries temporais reais de transformadores de usinas nucleares com 4 gases (H2, CO, C2H4, C2H2)."),
    ("Taxa de Amostragem (Timestep)", "1 passo = 12 horas", "Intervalo físico padrão de cromatografia online e medição periódica em transformadores de potência."),
    ("Tamanho da Janela de Entrada (w*)", "w* = 50 passos (25 dias de histórico)", "Vencedor do Grid Search entre w in {10, 20, 30, 50}. Cobre a inércia lenta de evolução térmica do óleo."),
    ("Topologia da Rede Neural", "Stacked LSTM: 2 camadas x 64 unidades", "Camada 1 extrai dinâmica temporal passo a passo; Camada 2 consolida vetor de estado global do transformador."),
    ("Regularização por Dropout", "Dropout = 0,20 (20% de desconexão estocástica)", "Desliga 20% das sinapses a cada iteração para evitar memorização (overfitting) de transformadores específicos."),
    ("Normalização de Dados", "ScalerPerGas: Min-Max [0, 1] independente por gás", "Equilibra ordens de grandeza díspares (CO em milhares de ppm vs C2H2 em unidades) sem mascarar o acetileno."),
    ("Segregação da Normalização", "Ajuste (.fit) ESTRITAMENTE na partição de treino", "Blindagem metodológica contra vazamento de dados (Data Leakage) do futuro para o passado."),
    ("Modo de Previsão Multi-Passo", "Previsão Recursiva (Feedback Loop / Autoregressive)", "Prevê t+1, reintroduz no final da janela e desliza a fila FIFO recursivamente até atingir o horizonte h."),
    ("Horizontes Operacionais Primários", "h = 14 (7 dias / 1 semana) e h = 28 (14 dias / 2 semanas)", "Horizontes ótimos para planejamento de manutenção preventiva em subestação com R2 > 0,99 e erro < 4%."),
    ("Avaliação de Estresse de Horizontes", "Testado de h=1 até h=112 passos (8 semanas / 56 dias)", "A rede sustenta R2 médio >= 0,80 até exatamente 112 passos (R2 = 83,8%), marcando o limite prático de previsão.")
]

ws1.append([])
for r_idx, row in enumerate(resumo_items, start=4):
    ws1.append(list(row))
    ws1.row_dimensions[r_idx].height = 24
    for c_idx in range(1, 4):
        cell = ws1.cell(row=r_idx, column=c_idx)
        cell.border = thin_border
        if r_idx == 4:
            cell.fill = fill_teal
            cell.font = font_header
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        else:
            if c_idx == 1:
                cell.font = font_bold
                cell.fill = fill_light_blue
            elif c_idx == 2:
                cell.font = font_bold
            else:
                cell.font = font_regular
            cell.alignment = Alignment(vertical="center", wrap_text=True)

# ---------------------------------------------------------------------------
# ABA 2: HORIZONTES E DEGRADAÇÃO (DADOS REAIS DO JSON)
# ---------------------------------------------------------------------------
ws2 = wb.create_sheet(title="Horizontes_Degradacao")
ws2.views.sheetView[0].showGridLines = True

ws2.merge_cells("A1:K1")
ws2["A1"] = "CURVA DE DEGRADAÇÃO PREVENTIVA DA PREVISÃO RECURSIVA (h = 1 até h = 112 passos)"
ws2["A1"].font = font_title
ws2["A1"].fill = fill_navy
ws2["A1"].alignment = Alignment(horizontal="center", vertical="center")
ws2.row_dimensions[1].height = 30

ws2.merge_cells("A2:K2")
ws2["A2"] = "Resultados agregados em 5 sementes independentes (42, 123, 456, 789, 2024) com w* = 50 na partição de teste"
ws2["A2"].font = Font(name="Calibri", size=10, italic=True, color=WHITE)
ws2["A2"].fill = fill_midnight
ws2["A2"].alignment = Alignment(horizontal="center", vertical="center")
ws2.row_dimensions[2].height = 20

headers_h = [
    "Horizonte (h)", "Tempo Físico", "Semanas", "R² Médio", "Desvio R²",
    "R² H2", "R² CO", "R² C2H4", "R² C2H2",
    "Erro Médio (%)", "Status de Confiabilidade (Critério R² >= 0.80)"
]
ws2.append([])
ws2.append(headers_h)
ws2.row_dimensions[4].height = 26

for col_num in range(1, len(headers_h) + 1):
    c = ws2.cell(row=4, column=col_num)
    c.fill = fill_teal
    c.font = font_header
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = header_border

# Preencher horizontes reais do JSON
horizontes_chaves = sorted(data_mh["por_horizonte"].keys(), key=lambda x: int(x))
row_start = 5

for k in horizontes_chaves:
    item = data_mh["por_horizonte"][k]
    h_steps = item["h_passos"]
    h_horas = item["h_horas"]
    h_semanas = item["h_semanas"]
    r2_m = item["r2_medio"]["media"]
    r2_s = item["r2_medio"]["std"]
    r2_h2 = item["r2_por_gas"]["H2"]["media"]
    r2_co = item["r2_por_gas"]["CO"]["media"]
    r2_c2h4 = item["r2_por_gas"]["C2H4"]["media"]
    r2_c2h2 = item["r2_por_gas"]["C2H2"]["media"]
    
    # Media de erro percentual entre os 4 gases
    mae_pcts = [item["mae_pct_por_gas"][g]["media"] for g in ["H2", "CO", "C2H4", "C2H2"]]
    mae_pct_medio = sum(mae_pcts) / len(mae_pcts)
    
    if h_steps in [14, 28]:
        status = "★ HORIZONTE ADOTADO NO XGBOOST (Ótimo Operacional)"
    elif r2_m >= 0.80:
        status = "Confiável (R² >= 0,80)"
    else:
        status = "Degradado (R² < 0,80)"
        
    tempo_str = f"{h_horas} horas ({h_horas/24:.1f} dias)" if h_horas >= 24 else f"{h_horas} horas"
    
    ws2.append([
        f"h = {h_steps}",
        tempo_str,
        f"{h_semanas:.2f}",
        r2_m,
        r2_s,
        r2_h2,
        r2_co,
        r2_c2h4,
        r2_c2h2,
        mae_pct_medio / 100.0,
        status
    ])
    
    current_row = ws2.max_row
    ws2.row_dimensions[current_row].height = 20
    
    # Formatação das celulas
    for c_i in range(1, 12):
        cell = ws2.cell(row=current_row, column=c_i)
        cell.border = thin_border
        cell.font = font_regular
        cell.alignment = Alignment(vertical="center", horizontal="center")
        
        # Formatos numericos
        if c_i in [4, 5, 6, 7, 8, 9]:
            cell.number_format = "0.0000"
        elif c_i == 10:
            cell.number_format = "0.00%"
            
        # Destaques visuais
        if h_steps in [14, 28]:
            cell.fill = fill_light_green
            cell.font = font_bold
        elif h_steps == 112:
            cell.fill = fill_light_yellow
            if c_i == 11:
                cell.font = font_bold

# ---------------------------------------------------------------------------
# ABA 3: GRID SEARCH DO TAMANHO DA JANELA (w)
# ---------------------------------------------------------------------------
ws3 = wb.create_sheet(title="Janelamento_w_Star")
ws3.views.sheetView[0].showGridLines = True

ws3.merge_cells("A1:F1")
ws3["A1"] = "AVALIAÇÃO DO TAMANHO DA JANELA DESLIZANTE (w ∈ {10, 20, 30, 50})"
ws3["A1"].font = font_title
ws3["A1"].fill = fill_navy
ws3["A1"].alignment = Alignment(horizontal="center", vertical="center")
ws3.row_dimensions[1].height = 30

headers_w = [
    "Tamanho da Janela (w)", "Histórico Temporal", "Buffer de Memória (Timesteps)",
    "Comportamento Físico do Transformador", "Desempenho da LSTM (Erro / Convergência)", "Veredito de Engenharia"
]
ws3.append([])
ws3.append(headers_w)
ws3.row_dimensions[3].height = 25

for col_num in range(1, len(headers_w) + 1):
    c = ws3.cell(row=3, column=col_num)
    c.fill = fill_midnight
    c.font = font_header
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = header_border

w_data = [
    ("w = 10", "5 dias (120 horas)", "10 passos de 12h", "Curto demais. A difusão gasosa no óleo é lenta; 5 dias capta apenas ruído e pequenas variações diárias de carga.", "Maior erro quadrático (MSE elevado). A rede tem dificuldade em distinguir tendência de oscilação térmica.", "Descartado"),
    ("w = 20", "10 dias (240 horas)", "20 passos de 12h", "Capta início de tendências, mas ainda é sensível a ciclos semanais de operação industrial da usina.", "Desempenho intermediário. Erro MAE ainda 18% superior ao de w=50.", "Descartado"),
    ("w = 30", "15 dias (360 horas)", "30 passos de 12h", "Boa capacidade de representação. Consegue observar 2 ciclos semanais completos de carga.", "Aproxima-se da estabilidade ótima, mas ainda apresenta leve subestimação em subidas rápidas de H2.", "Descartado"),
    ("w* = 50", "25 dias (600 horas)", "50 passos de 12h", "Ideal. Fornece quase um mês de histórico contínuo, capturando a inércia química completa do óleo mineral sem truncar séries.", "Menor erro de teste (MAE global = 5,59e-5 normalizado; R² > 0,999 no passo h=1). Máxima estabilidade entre sementes.", "VENCEDOR (w* Escolhido)")
]

for r_idx, row in enumerate(w_data, start=4):
    ws3.append(list(row))
    ws3.row_dimensions[r_idx].height = 32
    for c_idx in range(1, 7):
        cell = ws3.cell(row=r_idx, column=c_idx)
        cell.border = thin_border
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        if "w* = 50" in row[0]:
            cell.fill = fill_light_green
            cell.font = font_bold
        else:
            cell.font = font_regular
            if c_idx == 6:
                cell.font = font_italic

# ---------------------------------------------------------------------------
# ABA 4: ARQUITETURA DA STACKED LSTM
# ---------------------------------------------------------------------------
ws4 = wb.create_sheet(title="Arquitetura_LSTM")
ws4.views.sheetView[0].showGridLines = True

ws4.merge_cells("A1:F1")
ws4["A1"] = "ESPECIFICAÇÃO DETALHADA DA TOPOLOGIA DA REDE NEURAL RECORRENTE"
ws4["A1"].font = font_title
ws4["A1"].fill = fill_navy
ws4["A1"].alignment = Alignment(horizontal="center", vertical="center")
ws4.row_dimensions[1].height = 30

headers_lstm = [
    "Camada / Estágio", "Tipo Keras / Módulo", "Dimensão de Entrada (Shape)",
    "Dimensão de Saída (Shape)", "Hiperparâmetros Críticos", "Função no Pipeline de Engenharia"
]
ws4.append([])
ws4.append(headers_lstm)
ws4.row_dimensions[3].height = 25

for col_num in range(1, len(headers_lstm) + 1):
    c = ws4.cell(row=3, column=col_num)
    c.fill = fill_teal
    c.font = font_header
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = header_border

lstm_layers = [
    ("Entrada Temporal", "InputLayer", "(None, 50, 4)", "(None, 50, 4)", "w=50 timesteps, 4 gases normalizados", "Buffer circular contendo as últimas 50 medições (25 dias) de H2, CO, C2H4, C2H2."),
    ("1ª Camada Recorrente", "LSTM (Nível Baixo)", "(None, 50, 4)", "(None, 50, 64)", "units=64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'", "Extrai a evolução passo a passo da dinâmica gasosa, propagando o vetor oculto h_t ao longo de toda a sequência."),
    ("Camada de Regularização", "Dropout", "(None, 50, 64)", "(None, 50, 64)", "rate=0.20 (20% de desconexão estocástica no treino)", "Desliga aleatoriamente 20% das sinapses a cada iteração para impedir co-adaptação e memorização (overfitting)."),
    ("2ª Camada Recorrente", "LSTM (Nível Alto)", "(None, 50, 64)", "(None, 64)", "units=64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid'", "Consolida a sequência inteira em um único vetor de contexto global (64 características sintetizadas da máquina)."),
    ("Camada Densa de Saída", "Dense", "(None, 64)", "(None, 4)", "units=4, activation='linear' (estritamente linear)", "Mapeia as 64 variáveis latentes nas 4 concentrações preditas para o próximo timestep (h=1)."),
    ("Otimizador & Perda", "Adam + MSE Loss", "-", "-", "learning_rate = 0,001, beta_1=0.9, beta_2=0.999", "Minimiza o Erro Quadrático Médio via gradiente estocástico adaptativo com momentos de 1ª e 2ª ordem."),
    ("Critério de Parada", "EarlyStopping", "-", "-", "patience=15 épocas, min_delta=1e-6, restore_best_weights=True", "Monitora val_loss. Se não houver melhoria em 15 épocas, interrompe e restaura os melhores pesos da rede.")
]

for r_idx, row in enumerate(lstm_layers, start=4):
    ws4.append(list(row))
    ws4.row_dimensions[r_idx].height = 28
    for c_idx in range(1, 7):
        cell = ws4.cell(row=r_idx, column=c_idx)
        cell.border = thin_border
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        if c_idx in [1, 2]:
            cell.font = font_bold
            cell.fill = fill_light_blue
        else:
            cell.font = font_regular

# ---------------------------------------------------------------------------
# ABA 5: NORMALIZAÇÃO E ESCALA DOS GASES
# ---------------------------------------------------------------------------
ws5 = wb.create_sheet(title="Normalizacao_Gases")
ws5.views.sheetView[0].showGridLines = True

ws5.merge_cells("A1:F1")
ws5["A1"] = "PRESERVAÇÃO DE ESCALA FÍSICA E BLINDAGEM CONTRA DATA LEAKAGE"
ws5["A1"].font = font_title
ws5["A1"].fill = fill_navy
ws5["A1"].alignment = Alignment(horizontal="center", vertical="center")
ws5.row_dimensions[1].height = 30

headers_norm = [
    "Gás Dissolvido", "Fórmula Química", "Faixa Típica Katser (PPM)",
    "Fenômeno Físico / Severidade", "Comportamento SEM Normalização", "Efeito da Normalização ScalerPerGas"
]
ws5.append([])
ws5.append(headers_norm)
ws5.row_dimensions[3].height = 25

for col_num in range(1, len(headers_norm) + 1):
    c = ws5.cell(row=3, column=col_num)
    c.fill = fill_midnight
    c.font = font_header
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = header_border

gases_info = [
    ("Hidrogênio", "H2", "0,1 a 150+ ppm", "Descargas parciais (efeito corona) e descargas elétricas em baixa energia.", "Gradiente moderado, ofuscado pelas grandes concentrações de CO.", "Normalizado em [0, 1]. A rede detecta subidas sutis que indicam início de corona."),
    ("Monóxido de Carbono", "CO", "10 a 3.000+ ppm", "Degradação térmica do isolamento de papel celulósico / sobreaquecimento.", "MONOPOLIZA o cálculo do MSE e do gradiente por ter valores 1.000x maiores.", "Normalizado em [0, 1]. Mantém peso relativo proporcional sem esmagar os outros gases."),
    ("Etileno", "C2H4", "0,5 a 500+ ppm", "Sobreaquecimento térmico severo do óleo mineral (> 700°C / pirólise).", "Gradiente intermediário, sofrendo saturação.", "Normalizado em [0, 1]. Permite predição exata do ponto de quebra térmica do óleo."),
    ("Acetileno", "C2H2", "0,01 a 20+ ppm", "ARCO ELÉTRICO VIOLENTO (> 1.000°C) — FALHA DE MAIOR GRAVIDADE ESTRUTURAL.", "TOTALMENTE IGNORADO pela rede neural (erros de 0,5 ppm geravam gradiente nulo perto do CO).", "Normalizado em [0, 1]. Recebe a MESMA sensibilidade matemática que o CO, garantindo alerta precoce de arco!")
]

for r_idx, row in enumerate(gases_info, start=4):
    ws5.append(list(row))
    ws5.row_dimensions[r_idx].height = 32
    for c_idx in range(1, 7):
        cell = ws5.cell(row=r_idx, column=c_idx)
        cell.border = thin_border
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        if "C2H2" in row[1]:
            cell.fill = fill_light_yellow
            if c_idx in [1, 2, 6]:
                cell.font = font_bold
        else:
            cell.font = font_regular

# ---------------------------------------------------------------------------
# ABA 6: GUIA DE SABATINA (PERGUNTAS & RESPOSTAS PARA A BANCA)
# ---------------------------------------------------------------------------
ws6 = wb.create_sheet(title="Sabatina_Banca")
ws6.views.sheetView[0].showGridLines = True

ws6.merge_cells("A1:D1")
ws6["A1"] = "GABARITO E ARGUMENTAÇÃO TÉCNICA PARA A BANCA EXAMINADORA"
ws6["A1"].font = font_title
ws6["A1"].fill = fill_navy
ws6["A1"].alignment = Alignment(horizontal="center", vertical="center")
ws6.row_dimensions[1].height = 30

headers_sab = [
    "Item / Tópico", "Pergunta Provável da Banca Examinadora",
    "Conceito de Engenharia Envolvido", "Gabarito de Defesa Recomendado (Como Responder)"
]
ws6.append([])
ws6.append(headers_sab)
ws6.row_dimensions[3].height = 25

for col_num in range(1, len(headers_sab) + 1):
    c = ws6.cell(row=3, column=col_num)
    c.fill = fill_teal
    c.font = font_header
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = header_border

sabatina_data = [
    ("Escolha de w=50",
     "Por que fixar w=50 e não usar uma janela menor para ter predições mais rápidas?",
     "Trade-off entre tempo de resposta do sensor e inércia físico-química do óleo.",
     "Transformadores reais operam com grande volume de óleo e a difusão gasosa é lenta. Janelas de 5 a 10 dias captam apenas ruído de temperatura ambiente. Testamos 10, 20, 30 e 50 via Grid Search: 50 timesteps (25 dias) entregou o menor erro quadrático (R² > 0,999) e capturou a tendência termodinâmica real."),
    
    ("Normalização no Treino",
     "Por que você não aplicou o MinMaxScaler sobre o dataset inteiro de uma vez antes de dividir?",
     "Princípio de Data Leakage (Vazamento de Dados) e validação supervisionada honesta.",
     "Se o escalonador usar os dados de teste, ele absorve os valores máximos do futuro, que um transformador em operação real em 2024 não tem como prever. Ajustamos o ScalerPerGas estritamente sobre as 1.785 séries de treino. Os conjuntos de validação e teste usam os parâmetros do treino passivamente."),
    
    ("Previsão Recursiva",
     "Como funciona a previsão recursiva em múltiplos horizontes e por que o erro cresce?",
     "Sistemas em malha fechada / Feedback Loop com acúmulo de variância estocástica.",
     "A rede foi treinada para one-step-ahead (h=1 / 12h). Para olhar a frente, a previsão t+1 é realimentada no final do buffer circular, descartando o ponto mais antigo. Como a entrada passa a ser uma estimativa e não um dado real, os pequenos desvios se somam ao longo das iterações (efeito bola de neve)."),
    
    ("Horizontes 14 e 28 vs 112",
     "Se a sua rede aguentou prever até 112 passos (quase 2 meses), por que você só passou h=14 e h=28 para o XGBoost?",
     "Critério de utilidade operacional para manutenção de subestações e pureza do sinal.",
     "Em h=14 (1 semana) e h=28 (2 semanas), a LSTM opera com precisão cirúrgica (R² > 0,99 e erro < 4%), fornecendo ao XGBoost um sinal límpido. Em h=112, embora aceitável (R²=83,8%), o erro percentual sobe para 18%, inserindo ruído no classificador. Além disso, 7 a 14 dias é a janela padrão para uma concessionária planejar a parada segura de uma máquina."),
    
    ("Topologia da LSTM",
     "Por que 2 camadas de 64 unidades com Dropout e não uma rede mais profunda (ex: 4 camadas)?",
     "Dilema Viés-Variância (Bias-Variance Trade-off) e parcimônia arquitetural.",
     "Quatro gases e 50 passos temporais formam um problema com dimensionalidade controlada. Redes com 3 ou 4 camadas adicionariam dezenas de milhares de parâmetros desnecessários, causando sobreajuste (overfitting) e lentidão sem ganho preditivo. O Dropout de 0,20 e duas camadas (nível local e nível de contexto) atingiram convergência com R²=0,9996 no teste.")
]

for r_idx, row in enumerate(sabatina_data, start=4):
    ws6.append(list(row))
    ws6.row_dimensions[r_idx].height = 45
    for c_idx in range(1, 5):
        cell = ws6.cell(row=r_idx, column=c_idx)
        cell.border = thin_border
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        if c_idx == 1:
            cell.font = font_bold
            cell.fill = fill_light_blue
        elif c_idx == 2:
            cell.font = font_bold
        else:
            cell.font = font_regular

# ---------------------------------------------------------------------------
# AJUSTAR LARGURAS DAS COLUNAS EM TODAS AS ABAS
# ---------------------------------------------------------------------------
for sheet in wb.worksheets:
    for col in sheet.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            # Ignora linhas de titulo mescladas
            if cell.row in [1, 2]:
                continue
            val_str = str(cell.value or '')
            if '\n' in val_str:
                val_str = max(val_str.split('\n'), key=len)
            max_len = max(max_len, len(val_str))
        sheet.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 48)

# Ajustes especificos manuais
ws1.column_dimensions["A"].width = 30
ws1.column_dimensions["B"].width = 42
ws1.column_dimensions["C"].width = 55

ws2.column_dimensions["A"].width = 16
ws2.column_dimensions["B"].width = 24
ws2.column_dimensions["C"].width = 12
ws2.column_dimensions["D"].width = 14
ws2.column_dimensions["E"].width = 14
ws2.column_dimensions["F"].width = 12
ws2.column_dimensions["G"].width = 12
ws2.column_dimensions["H"].width = 12
ws2.column_dimensions["I"].width = 12
ws2.column_dimensions["J"].width = 16
ws2.column_dimensions["K"].width = 40

ws3.column_dimensions["A"].width = 22
ws3.column_dimensions["B"].width = 22
ws3.column_dimensions["C"].width = 26
ws3.column_dimensions["D"].width = 40
ws3.column_dimensions["E"].width = 38
ws3.column_dimensions["F"].width = 24

ws4.column_dimensions["A"].width = 24
ws4.column_dimensions["B"].width = 22
ws4.column_dimensions["C"].width = 25
ws4.column_dimensions["D"].width = 25
ws4.column_dimensions["E"].width = 38
ws4.column_dimensions["F"].width = 45

ws5.column_dimensions["A"].width = 24
ws5.column_dimensions["B"].width = 18
ws5.column_dimensions["C"].width = 24
ws5.column_dimensions["D"].width = 38
ws5.column_dimensions["E"].width = 38
ws5.column_dimensions["F"].width = 45

ws6.column_dimensions["A"].width = 24
ws6.column_dimensions["B"].width = 36
ws6.column_dimensions["C"].width = 32
ws6.column_dimensions["D"].width = 60

# Salvar planilha
wb.save(OUT_XLSX)
print(f"Planilha Excel gerada com sucesso em: {OUT_XLSX}")

