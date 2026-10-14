# Bases de Dados do Projeto

Este diretório descreve e armazena os conjuntos de dados (datasets) utilizados para o treinamento, validação e teste dos modelos de aprendizado de máquina desenvolvidos no projeto.

Devido à natureza híbrida da arquitetura proposta no trabalho, as bases de dados foram separadas para atender a duas abordagens metodológicas distintas: **Previsão Temporal** (focada na evolução dos gases ao longo do tempo) e **Classificação Diagnóstica** (focada no diagnóstico preciso das falhas baseado em relações físico-químicas).

---

## 1. Módulo de Previsão Temporal (Estágio 1 — LSTM)

**Fonte:** Katser et al. (2022) - *Machine Learning Methods for Anomaly Detection in Nuclear Power Plant Power Transformers* (arXiv:2211.11013, disponibilizado via Kaggle).
**Diretório de Origem:** `data/external/katser_npp_transformers/`

Este dataset é composto por medições reais de gases dissolvidos em óleo (DGA) de transformadores de usinas nucleares. Ele foi selecionado por fornecer séries temporais longas com amostragem regular, permitindo o treinamento do modelo preditivo (LSTM) para aprender a dinâmica de evolução dos gases.

*   **Tamanho:** 3.000 séries temporais reais (divididas pelos autores em 2.100 para treino e 900 para teste).
*   **Dimensão da Série:** Cada série possui 420 pontos sequenciais.
*   **Taxa de Amostragem:** Dados coletados a cada 12 horas, totalizando aproximadamente 210 dias de monitoramento histórico por série.
*   **Gases Monitorados (Features):** Concentrações de H2, CO, C2H4, e C2H2.
*   **Classes Presentes (4 classes):** Estado Normal, Descarga Parcial (PD), Descarga de Baixa Energia (D1), e Sobreaquecimento de Baixa Temperatura (T1).

---

## 2. Módulo de Classificação Diagnóstica (Estágio 2 — IGA-XGBoost)

**Fonte:** Compilação de múltiplas fontes da literatura e do IEEE DataPort (Enwen Li, 2019, DOI: 10.21227/h8g0-8z59), mantida originalmente no repositório `alan-456/transformer-fault-dataset`.
**Diretório de Origem:** `data/external/dga_compiled_zh/`

Este dataset estático de DGA foi escolhido por possuir um volume elevado de amostras (incomum na literatura da área) e ser rigorosamente distribuído entre todas as sete classes de falhas padronizadas (taxonomia compatível com as normas IEC 60599). Ele serve como base de conhecimento para o classificador final (XGBoost), focado em fornecer um diagnóstico eletrotérmico completo.

*   **Tamanho Inicial:** 2.321 amostras (sujeitas a deduplicação e limpeza durante o pré-processamento).
*   **Gases Monitorados:** Concentrações de H2, CO, CH4, C2H4, C2H6, C2H2, e CO2. A presença de todos os gases combustíveis chave permite o cálculo de todas as razões da norma IEC e plotagem no Triângulo de Duval.
*   **Taxonomia (7 Classes de Falha):** 
    1. Normal
    2. Descarga Parcial (PD)
    3. Descarga de Baixa Energia (D1)
    4. Descarga de Alta Energia (D2)
    5. Sobreaquecimento Baixo (T1 - < 300 °C)
    6. Sobreaquecimento Médio (T2 - 300 a 700 °C)
    7. Sobreaquecimento Alto (T3 - > 700 °C)

---

## 3. Estrutura de Diretórios e Reprodutibilidade

- `data/external/`: Armazena os dados brutos e intocados exatamente da forma como foram obtidos das fontes primárias. Nenhuma modificação é feita diretamente nestes arquivos para garantir reprodutibilidade.
- `data/processed/`: Armazena configurações geradas pelo sistema, *scalers* salvos (ex: pesos de normalização), splits de dados processados e os melhores hiperparâmetros encontrados pelos algoritmos de busca (ex: configurações da LSTM).

> **Nota sobre o Controle de Versão:** Todos os arquivos de dados brutos encontram-se versionados neste repositório
