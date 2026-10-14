# Predição Dinâmica de Falhas em Transformadores

**Repositório Oficial do Trabalho de Conclusão de Curso (TCC) - UNISINOS**

Este repositório contém a implementação completa do modelo híbrido **LSTM + IGA-XGBoost** para predição proativa e diagnóstico de falhas incipientes em transformadores de potência, utilizando Análise de Gases Dissolvidos (DGA).

**Autor:** Ândrio Gabriel Epping  
**Orientadora:** Profa. Me. Janaína Conceição Sutil Lemos  

---

## Sobre o Projeto

A manutenção preditiva de transformadores de potência é crítica para a segurança e estabilidade do sistema elétrico. Os métodos tradicionais de DGA (Análise de Gases Dissolvidos) são amplamente utilizados, mas possuem limitações na identificação precoce e na modelagem da evolução temporal das falhas.

Este projeto propõe uma arquitetura de aprendizado de máquina em duas etapas:
1. **Previsão Temporal (LSTM):** Uma rede neural recorrente prevê a evolução das concentrações de gases combustíveis (H2, CO, C2H4, C2H2) no futuro, analisando o histórico temporal de degradação.
2. **Diagnóstico (IGA-XGBoost):** Um classificador otimizado por Algoritmo Genético Imune recebe as previsões temporais, combinadas com conhecimento físico (Razões da IEC e métodos clássicos), para diagnosticar precocemente e com alta acurácia o tipo de falha que ocorrerá.

---

##  Estrutura do Repositório

O projeto foi estruturado da seguinte forma:

- `data/`: Datasets originais e processados. Consulte o arquivo `data/README.md` para o rigor metodológico de escolha de cada base.
- `notebooks/`: Análises exploratórias e documentação interativa em Jupyter.
- `src/`: Código-fonte modular da aplicação.
  - `preprocessing/`: Pipelines de limpeza, normalização e extração de janelas deslizantes temporais.
  - `lstm/`: Arquitetura, treinamento de multi-horizontes e *early stopping* da rede LSTM.
  - `iga_xgboost/`: Implementação do Algoritmo Genético acoplado ao classificador XGBoost.
  - `evaluation/`: Protocolos rigorosos de métricas de desempenho e validação.
- `reports/`: Relatórios gerados, figuras, gráficos de resultados e protocolos acadêmicos (veja `validation_checklist.md`).
- `tests/`: Scripts de testes automatizados de integridade dos dados e das funções.

---

## Bases de Dados

Foram utilizadas duas bases de dados reais:
1. **Base Temporal (Katser et al., 2022):** 3.000 séries temporais (medidas a cada 12h) usadas exclusivamente para ensinar a LSTM a prever a curva futura dos gases.
2. **Base Diagnóstica (IEC 60599 / China):** 2.321 amostras estáticas cobrindo perfeitamente as 7 classes padronizadas de falhas térmicas e elétricas, usadas para treinar o classificador XGBoost.

> **Documentação de Dados:** O processo completo de busca, validação contra vazamento de dados, deduplicação e análise exploratória de ambas as bases está registrado em [`data/README.md`](data/README.md).

---

## Instalação e Reprodução

Para reproduzir os treinamentos e avaliações, configure seu ambiente virtual Python e instale as dependências:

```bash
# 1. Criar o ambiente virtual
python -m venv .venv

# 2. Ativar o ambiente virtual
# No Windows (PowerShell):
.\.venv\Scripts\activate
# No Linux/macOS:
source .venv/bin/activate

# 3. Instalar pacotes necessários
pip install -r requirements.txt
```

*(O código foi construído com a variável `RANDOM_SEED = 42` globalizada no arquivo `src/config.py`, garantindo total reprodutibilidade dos resultados e métricas em qualquer ambiente de execução).*
