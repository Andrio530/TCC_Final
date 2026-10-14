# Predição Dinâmica de Falhas em Transformadores de Potência

**Abordagem Híbrida de Previsão Temporal (LSTM) e Classificação Otimizada (IGA-XGBoost)**

Trabalho de Conclusão de Curso (TCC II) apresentado ao Curso de Engenharia da Computação da Universidade do Vale do Rio dos Sinos (UNISINOS).

**Autor:** Ândrio Gabriel Epping  
**Orientadora:** Profa. Me. Janaína Conceição Sutil Lemos  
**Ano:** 2026  

---

## 📌 Visão Geral do Repositório

Este repositório unifica a documentação acadêmica e o pipeline computacional completo de experimentos para diagnóstico e predição de falhas incipientes em transformadores de potência imersos em óleo mineral via Análise de Gases Dissolvidos (DGA).

O projeto é estruturado em dois diretórios principais:

1. **`tcc-unisinos/`**: Monografia acadêmica em LaTeX estruturada em conformidade com as normas ABNT e o padrão institucional da UNISINOS. Contém a totalidade dos cinco capítulos redigidos, arquivos pré e pós-textuais, bibliografia completa (`referencias.bib`), figuras geradas e o documento final compilado (`main.pdf`).
2. **`tcc-transformer-fault/`**: Pipeline experimental completo em Python, compreendendo scripts de saneamento e auditoria de dados, modelagem temporal por redes neurais recorrentes LSTM, otimização hiperparamétrica e seleção de atributos via Algoritmo Genético Melhorado (IGA com DEAP), classificação por ensembles de árvores XGBoost, testes de hipótese por bootstrap pareado e validação cruzada repetida, e rotinas de interpretabilidade físico-química pós-hoc via TreeSHAP.

---

## 🔬 Estrutura Metodológica em Dois Eixos

Devido à assimetria das bases industriais públicas disponíveis, a pesquisa foi formalmente dividida em dois eixos operacionais:

* **Track A (Séries Temporais Industriais):** Avalia 3.000 séries temporais multivariadas amostradas a cada 12 horas (Katser et al., 2022; Skoltech/Rosatom) para 4 gases ($\text{H}_2$, $\text{CO}$, $\text{C}_2\text{H}_4$, $\text{C}_2\text{H}_2$) em 4 classes operacionais (Normal, PD, D1, T1). Emprega LSTM empilhada ($w=50$, $h=14, 28$) acoplada a XGBoost sob teste formal de ablação da hipótese temporal.
* **Track B (Diagnóstico Multiclasse Estático):** Avalia 1.972 amostras únicas de cromatografia de campo após deduplicação mandatória de 349 registros redundantes (15%), cobrindo 5 gases ($\text{H}_2$, $\text{CH}_4$, $\text{C}_2\text{H}_6$, $\text{C}_2\text{H}_4$, $\text{C}_2\text{H}_2$) e 7 classes da IEC 60599 (Normal, PD, D1, D2, T1, T2, T3). Confronta o modelo IGA-XGBoost contra o XGBoost padrão, o modelo de Wu et al. (2021) e as regras normativas clássicas da IEC 60599 e do Triângulo 1 de Duval.

---

## 📁 Estrutura de Diretórios

```text
TCC_Final/
├── README.md                          # Apresentação consolidada do repositório
├── .gitignore                         # Regras de exclusão unificadas (Python e LaTeX)
│
├── tcc-unisinos/                      # Monografia em LaTeX (UNISINOS)
│   ├── main.tex                       # Documento mestre LaTeX
│   ├── main.pdf                       # Monografia completa compilada (69 páginas)
│   ├── pretextual.tex                 # Capa, folha de rosto, resumo e abstract
│   ├── introducao.tex                 # Capítulo 1: Introdução
│   ├── fundamentacao.tex              # Capítulo 2: Fundamentação Teórica
│   ├── metodologia.tex                # Capítulo 3: Metodologia
│   ├── resultados.tex                 # Capítulo 4: Resultados e Discussão
│   ├── conclusao.tex                  # Capítulo 5: Conclusão
│   ├── referencias.bib                # Base de dados bibliográfica BibTeX
│   ├── plano_escrita_tcc_final.md     # Plano estratégico de escrita e defesa
│   └── figuras/                       # Ilustrações e gráficos (incluindo fase 6 XAI)
│
└── tcc-transformer-fault/             # Pipeline computacional dos experimentos
    ├── data/                          # Dados de DGA (raw, processed, train/val/test)
    ├── src/                           # Código-fonte Python
    │   ├── data/                      # Pré-processamento, deduplicação e normalização
    │   ├── models/                    # Modelagem LSTM, XGBoost, IGA e baselines
    │   ├── evaluation/                # Métricas, bootstrap pareado e validação cruzada
    │   └── explainability/            # Rotinas de TreeSHAP e matrizes de interação
    ├── reports/                       # Relatórios de métricas (JSON) e figuras (PNG/PDF)
    └── tests/                         # Testes automatizados unitários e de integração
```

---

## ⚙️ Instruções de Uso e Reprodução

### 1. Compilação da Monografia (LaTeX)

Para compilar o documento mestre a partir do diretório `tcc-unisinos/`:

```bash
cd tcc-unisinos
latexmk -pdf -interaction=nonstopmode main.tex
```

O arquivo `main.pdf` resultante será atualizado diretamente no diretório raiz de `tcc-unisinos/`.

### 2. Execução dos Experimentos em Python

Recomenda-se a utilização de ambiente virtual Python (versão $\ge 3.10$):

```bash
cd tcc-transformer-fault
python -m venv .venv
# Ativação no Windows:
.venv\Scripts\activate
# Instalação dos requisitos:
pip install -r requirements.txt
```

Para rodar os testes unitários e de validação normativa:

```bash
pytest tests/
```

---

## 📜 Licença e Citação

Este material foi desenvolvido para fins acadêmicos e científicos no âmbito do curso de Engenharia da Computação da Universidade do Vale do Rio dos Sinos (UNISINOS).
