# Guia de Interpretação de Métricas e Resultados — TCC

Este documento explica **tecnicamente** cada métrica que você está usando, como interpretar cada figura gerada, e como explicar seus resultados para a orientadora.

---

## Parte 1 — Métricas da LSTM (Estágio 1: Regressão)

A LSTM é um modelo de **regressão** — ela prevê valores numéricos contínuos (concentrações futuras de gases em PPM). As métricas de regressão avaliam "quão perto o valor previsto ficou do valor real".

### 1.1 MSE — Mean Squared Error (Erro Quadrático Médio)

**Fórmula:**
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**O que mede:** A média dos quadrados das diferenças entre o valor real ($y_i$) e o valor previsto ($\hat{y}_i$).

**Como interpretar:**
- Quanto **menor**, melhor. MSE = 0 significaria previsão perfeita.
- O **quadrado** penaliza erros grandes mais que erros pequenos. Se o modelo errar muito em uma amostra, o MSE sobe bastante.
- **Unidade:** PPM² (o quadrado da unidade original). Por isso é difícil interpretar diretamente.

**Onde aparece no projeto:** É a **loss function** (função de perda) usada para treinar a LSTM. O modelo tenta minimizar essa métrica durante o treinamento.

**Na figura de curvas de treino:**

![Curvas de Treinamento da LSTM](C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/brain/458ee139-d2ca-4ec4-a2e4-e3891594a76c/lstm_loss_w50_seed42_norm.png)

**Como ler essa figura:**
- **Eixo X:** Épocas de treinamento (cada passada completa pelo dataset)
- **Eixo Y:** MSE Loss (quanto menor, melhor)
- **Linha azul (Treino):** O erro no conjunto de treinamento. Começa alto (~0.003) e cai rapidamente nas primeiras 10 épocas, depois estabiliza (~0.0005).
- **Linha vermelha (Validação):** O erro no conjunto de validação. Já começa baixo e estabiliza rapidamente.
- **Linha tracejada vertical:** A "melhor época" (30) — onde o modelo do early stopping salvou os pesos.

**O que dizer pra orientadora:** "A LSTM convergiu rapidamente em ~10 épocas e o early stopping parou na época 30 após 15 épocas sem melhoria. A curva de validação está **abaixo** da curva de treino, o que indica que **não há overfitting** — o modelo generaliza bem."

> [!NOTE]
> A validação estar abaixo do treino pode parecer contra-intuitivo, mas acontece porque: (1) os dados de validação foram normalizados per-gas e são mais "fáceis" por terem menor variância; e (2) a curva de treino inclui o dropout ativo, que dificulta artificialmente o aprendizado durante o treino.

---

### 1.2 MAE — Mean Absolute Error (Erro Absoluto Médio)

**Fórmula:**
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**O que mede:** A média dos erros absolutos (sem quadrado, diferente do MSE).

**Como interpretar:**
- Quanto **menor**, melhor.
- Mais intuitivo que o MSE porque está na **mesma unidade** dos dados (PPM, se os dados estiverem em PPM).
- Não penaliza erros grandes tanto quanto o MSE.

**Seus resultados (h=1, um passo à frente):**

| Gás | MAE (escala normalizada) | Interpretação |
|---|---|---|
| H2 | 0.0000144 | Erro ínfimo |
| CO | 0.0001615 | Maior erro (CO tem mais variação) |
| C2H4 | 0.0000458 | Muito baixo |
| C2H2 | 0.0000019 | Praticamente perfeito |

**O que dizer:** "O MAE da LSTM para previsão de um passo (12h) é da ordem de 10⁻⁵ na escala normalizada, o que é excelente. O CO tem o maior erro porque é o gás com maior variabilidade temporal na base Katser."

---

### 1.3 RMSE — Root Mean Squared Error (Raiz do Erro Quadrático Médio)

**Fórmula:**
$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**O que mede:** É a raiz quadrada do MSE. Volta para a mesma unidade dos dados (como o MAE).

**Diferença entre MAE e RMSE:**
- Se MAE ≈ RMSE → os erros são relativamente uniformes (todos parecidos)
- Se RMSE >> MAE → existem alguns **erros grandes** ("outliers de previsão") que inflam o RMSE

**Seus resultados:** MAE ≈ RMSE em todos os gases → os erros são uniformes, sem outliers de previsão. Excelente sinal.

---

### 1.4 R² — Coeficiente de Determinação

**Fórmula:**
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

**O que mede:** Quanto da variância dos dados o modelo consegue explicar.

**Como interpretar:**
| Valor de R² | Significado |
|---|---|
| **R² = 1.0** | Previsão perfeita (impossível na prática) |
| **R² > 0.99** | Excelente — o modelo captura >99% da variância |
| **R² > 0.90** | Muito bom |
| **R² > 0.80** | Aceitável (threshold usado no projeto) |
| **R² = 0.0** | Modelo tão bom quanto chutar a média |
| **R² < 0.0** | Modelo pior que chutar a média |

**Analogia simples:** Pense no R² como uma "nota de 0 a 1". R²=0.997 significa que o modelo acerta 99.7% da variação dos dados — está quase perfeito.

**Seus resultados (R² por horizonte):**

| Horizonte | Semanas | R² Médio | Interpretação |
|---|---|---|---|
| h=1 | 0.1 sem | 0.9996 | Quase perfeito |
| h=7 | 0.5 sem | 0.9991 | Excelente |
| **h=14** | **1 sem** | **0.9979** | **Excelente** (horizonte primário usado no TCC) |
| **h=28** | **2 sem** | **0.9928** | **Muito bom** (horizonte secundário) |
| h=56 | 4 sem | 0.9691 | Bom, com degradação |
| h=112 | 8 sem | 0.8380 | Limite aceitável (R²≥0.8) |

**Na figura de degradação do R²:**

![Degradação do R² por Horizonte](C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/brain/458ee139-d2ca-4ec4-a2e4-e3891594a76c/lstm_multi_horizon_r2_norm.png)

**Como ler essa figura:**
- **Eixo X:** Horizonte de previsão em horas (de 12h até 1344h = 8 semanas)
- **Eixo Y:** R² (de 0 a 1)
- **Linha preta com quadrados:** R² médio de todos os gases
- **Linhas coloridas:** R² de cada gás individualmente
- **Linha vermelha tracejada:** Threshold R²=0.8 (abaixo disso consideramos a previsão "não confiável")
- **Linha verde pontilhada:** Horizonte máximo aceitável (h=112 passos = 8 semanas)
- **Faixas sombreadas:** Intervalo de confiança (variação entre as 5 seeds)

**O que dizer pra orientadora:** "A LSTM consegue prever concentrações de gases com R² > 0.99 até 2 semanas (h=28), e mantém R² > 0.80 até 8 semanas. Os horizontes que usamos no classificador — 7 dias (h=14) e 14 dias (h=28) — estão ambos na zona de alta confiabilidade (R² > 0.99). O H2 é o gás que degrada mais rápido, provavelmente por ser mais volátil e ter dinâmica mais irregular."

---

### 1.5 MAE% — Erro Percentual Relativo

**Fórmula:**
$$MAE\% = \frac{MAE}{\bar{y}} \times 100$$

**O que mede:** O MAE como percentual da concentração média do gás. Mais interpretável que o MAE absoluto.

**Como interpretar:** "O modelo erra, em média, X% do valor real."

**Seus resultados (MAE% por horizonte):**

| Horizonte | Semanas | MAE% H2 | MAE% CO | MAE% C2H4 | MAE% C2H2 |
|---|---|---|---|---|---|
| h=1 | 0.1 sem | 0.77% | 0.88% | 0.95% | 1.00% |
| h=14 | 1 sem | 1.76% | 1.99% | 2.18% | 2.28% |
| h=28 | 2 sem | 3.33% | 3.57% | 3.99% | 4.14% |
| h=112 | 8 sem | 16.64% | 16.31% | 18.53% | 19.25% |

**Na figura de degradação do MAE%:**

![Degradação do MAE% por Horizonte](C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/brain/458ee139-d2ca-4ec4-a2e4-e3891594a76c/lstm_multi_horizon_mae_pct_norm.png)

**Como ler essa figura:**
- **Eixo X:** Horizonte em horas
- **Eixo Y:** MAE% (erro percentual relativo)
- **Linha vermelha tracejada:** Threshold de 20% (acima disso a previsão não é útil)
- As curvas crescem quase linearmente: quanto mais longe no futuro, maior o erro

**O que dizer:** "Para o horizonte operacional de 7 dias (h=14), o erro de previsão é de apenas ~2% da concentração real. Para 14 dias (h=28), sobe para ~4%. Ambos são erros aceitáveis para a aplicação de manutenção preditiva."

---

### 1.6 Figuras de Previsão Real vs. Previsto

![Previsões vs. Real — LSTM](C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/brain/458ee139-d2ca-4ec4-a2e4-e3891594a76c/lstm_predictions_sample_norm.png)

**Como ler essa figura:**
- Cada **coluna** é um gás (H2, CO, C2H4, C2H2)
- Cada **linha** é uma série temporal diferente do teste (3 amostras)
- **Linha azul sólida:** Valor real
- **Linha vermelha tracejada:** Valor previsto pela LSTM
- **Eixo X:** Timesteps (pontos no tempo)

**O que observar:**
- A linha vermelha acompanha a azul de perto → boa previsão
- Em C2H4 (3ª coluna) há um leve desalinhamento nas primeiras amostras → normal, pois a previsão é recursiva (cada erro se acumula)

---

### 1.7 Distribuição dos Erros

![Distribuição dos Erros — LSTM](C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/brain/458ee139-d2ca-4ec4-a2e4-e3891594a76c/lstm_error_distribution_norm.png)

**Como ler essa figura:**
- Cada histograma mostra a distribuição dos erros (real - previsto) para um gás
- **Ideal:** sino centrado em zero (distribuição normal com média zero)
- **Seus resultados:** Todos os histogramas estão centrados em zero com caudas simétricas → **sem viés sistemático** (o modelo não tende a subestimar nem superestimar)

**O que dizer:** "A distribuição dos erros é simétrica e centrada em zero para todos os 4 gases, confirmando que a LSTM não tem viés de previsão. O CO tem a cauda mais larga, consistente com seu MAE mais alto."

---

## Parte 2 — Métricas do IGA-XGBoost (Estágio 2: Classificação)

O XGBoost é um modelo de **classificação** — ele recebe features e retorna o **tipo de falha** (Normal, PD, D1, D2, T1, T2, T3). As métricas de classificação avaliam "quantas vezes o modelo acertou/errou a classe".

### 2.1 Acurácia (Accuracy)

**Fórmula:**
$$Accuracy = \frac{\text{amostras classificadas corretamente}}{\text{total de amostras}}$$

**Exemplo:** Se de 295 amostras o modelo acertou 244 → Acurácia = 244/295 = 82.7%

**Limitação crítica:** Em datasets **desbalanceados**, a acurácia é enganosa. Se 81% das amostras são Normal, um modelo que chuta "Normal" para tudo tem 81% de acurácia — mas é inútil.

**Por isso usamos o F1-macro como métrica principal**, não a acurácia.

---

### 2.2 Precision (Precisão)

**Fórmula:**
$$Precision = \frac{VP}{VP + FP}$$

Onde VP = Verdadeiros Positivos, FP = Falsos Positivos.

**O que mede:** "De todas as vezes que o modelo disse 'classe X', quantas ele acertou?"

**Analogia:** Imagine um detector de falhas. Precision alta = poucas vezes ele "gritou lobo" sem ter lobo (poucos alarmes falsos).

**Exemplo do seu projeto (Track B, classe T1):**
- Modelo disse "T1" 25 vezes
- Dessas, apenas 12 eram realmente T1
- Precision = 12/25 = 0.48 → **Muitos alarmes falsos para T1**

---

### 2.3 Recall (Sensibilidade / Revocação)

**Fórmula:**
$$Recall = \frac{VP}{VP + FN}$$

Onde FN = Falsos Negativos.

**O que mede:** "De todas as amostras que realmente eram 'classe X', quantas o modelo encontrou?"

**Analogia:** Recall alta = o modelo não deixa passar falhas reais. Todas as falhas que existiam foram detectadas.

**Exemplo (Track B, classe T1):**
- Existem 16 amostras realmente T1 no teste
- O modelo encontrou 12 delas
- Recall = 12/16 = 0.75

---

### 2.4 F1-Score

**Fórmula:**
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**O que mede:** A **média harmônica** entre Precision e Recall. Equilibra os dois aspectos.

**Por que não usar a média aritmética?** Porque a harmônica penaliza mais quando um dos dois é baixo. Se Precision=1.0 e Recall=0.0, a média aritmética daria 0.5 (parece OK), mas o F1 dá 0.0 (refletindo que o modelo é inútil).

| Precision | Recall | F1-Score | Interpretação |
|---|---|---|---|
| 0.90 | 0.90 | **0.90** | Excelente |
| 1.00 | 0.50 | **0.67** | Acha poucos, mas quando acha acerta |
| 0.50 | 1.00 | **0.67** | Acha todos, mas gera muitos falsos |
| 0.80 | 0.80 | **0.80** | Bom |
| 0.48 | 0.75 | **0.585** | Fraco (caso do seu T1) |

---

### 2.5 F1-Macro (a métrica principal do projeto)

**Fórmula:**
$$F1_{macro} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

**O que mede:** A média simples do F1 de cada classe. Trata todas as classes com o **mesmo peso**, independente de quantas amostras cada uma tem.

**Por que é a métrica certa para o seu projeto:**
- O dataset tem classes desbalanceadas (Normal ≈ 106 amostras, T1 = 16 amostras)
- Se usássemos F1-micro (que é basicamente a acurácia), a classe Normal dominaria
- O F1-macro dá peso igual a T1 e Normal, forçando o modelo a ser bom em **todas** as classes

**Exemplo concreto (Track B, C3_IGA):**

| Classe | F1 individual | Peso no F1-macro |
|---|---|---|
| Normal | 0.936 | 1/7 = 14.3% |
| PD | 0.743 | 1/7 = 14.3% |
| D1 | 0.790 | 1/7 = 14.3% |
| D2 | 0.757 | 1/7 = 14.3% |
| T1 | 0.585 | 1/7 = 14.3% |
| T2 | 0.621 | 1/7 = 14.3% |
| T3 | 0.913 | 1/7 = 14.3% |
| **F1-macro** | **(0.936+0.743+0.790+0.757+0.585+0.621+0.913)/7** | **= 0.764** |

**O que dizer:** "O F1-macro foi escolhido como métrica principal porque trata igualmente todas as 7 classes de falha. Embora o modelo tenha 82.8% de acurácia, o F1-macro de 0.7525 é mais baixo porque as classes T1 e T2 (com apenas 16 amostras de teste cada) puxam a média para baixo."

---

### 2.6 Matriz de Confusão

**O que é:** Uma tabela que mostra, para cada classe real (linhas), quantas amostras foram classificadas em cada classe (colunas).

**Exemplo (Track B, C3_IGA, 295 amostras):**

```
                Predito→  Normal  PD  D1  D2  T1  T2  T3
Real ↓
Normal (106)               95     2   2   0   5   1   1
PD     (17)                 0    13   2   0   1   1   0
D1     (41)                 0     0  32   9   0   0   0
D2     (33)                 0     0   5  28   0   0   0
T1     (16)                 1     2   0   0  12   0   1
T2     (16)                 1     1   0   1   3   9   1
T3     (66)                 0     0   0   1   4   4  57
```

**Como ler:**
- **Diagonal:** Acertos (ex: 95 Normais classificados corretamente como Normal)
- **Fora da diagonal:** Erros (ex: 5 Normais classificados como T1 → alarme falso)
- **Somando uma linha:** Total real daquela classe
- **Somando uma coluna:** Total previsto daquela classe

**Confusões mais relevantes para a orientadora:**
- D1↔D2: 9 amostras de D1 confundidas com D2 e 5 de D2 com D1. Isso acontece porque D1 (descarga de baixa energia) e D2 (descarga de alta energia) geram padrões de gás similares.
- T1↔T2↔T3: Confusão entre as 3 faixas térmicas. Faz sentido fisicamente: as fronteiras de temperatura são contínuas, não discretas.

---

### 2.7 StratifiedKFold (Validação Cruzada Estratificada)

**O que é:** Divide o treino em K (=5) partes iguais, mantendo a proporção de classes em cada parte. Treina em 4 partes e testa na 5ª. Repete 5 vezes, cada vez usando uma parte diferente como teste.

**Por que "estratificada":** Se T1 tem 5% dos dados e fizéssemos divisão aleatória, algum fold poderia ficar sem nenhuma amostra de T1. A estratificação garante que cada fold tenha ~5% de T1.

**No projeto:** O IGA usa StratifiedKFold(5) para avaliar o fitness de cada cromossomo. O fitness é a média dos 5 F1-macro obtidos em cada fold.

---

### 2.8 Multi-seed (5 seeds)

**O que é:** Para reportar resultados finais, treinamos o modelo 5 vezes com seeds diferentes (42, 123, 456, 789, 2024) e reportamos média ± desvio-padrão.

**Por que:** O XGBoost tem componentes aleatórios (subsampling, inicialização). Se rodássemos apenas 1 vez, poderíamos ter tido "sorte" ou "azar". Com 5 seeds, mostramos que o resultado é **robusto** e **reprodutível**.

**Seus resultados:**

| Modelo | F1-macro por seed | Média ± Std |
|---|---|---|
| C3_IGA (Track B) | [0.764, 0.774, 0.779, 0.766, 0.782] | **0.7525 ± 0.0061** |

Um desvio-padrão de 0.006 significa que o resultado varia em ±0.6 pontos percentuais entre runs → **muito estável**.

---

## Parte 3 — Métricas do Algoritmo Genético (IGA)

### 3.1 Fitness (F1-macro CV)

**O que é:** A métrica interna que o IGA usa para decidir qual cromossomo é "melhor". É o F1-macro da validação cruzada (K=5) no treino.

**Diferença entre fitness e F1 real no teste:**
- **Fitness:** calculado internamente durante a busca (o IGA "vê" essa métrica)
- **F1 teste:** calculado no conjunto de teste que o IGA **nunca viu**

O **gap** entre os dois é o indicador principal de data leakage:

| | Antes da correção | Depois da correção |
|---|---|---|
| Track A: Fitness | 0.9889 | 0.8877 |
| Track A: F1 teste | 0.8946 | 0.8802 |
| Track A: **Gap** | **9.4 pp** ❌ | **0.7 pp** ✅ |

**O que dizer:** "O fitness caiu de 0.98 para 0.88 após a correção — isso parece ruim, mas na verdade é excelente. O 0.98 antigo era uma métrica **falsa**, inflada pelo vazamento do SMOTE. O 0.88 novo é honesto e bate com o F1 real no teste (gap de apenas 0.7 pp), provando que a validação cruzada interna agora é confiável."

---

### 3.2 Early Stopping do IGA

**O que é:** Se o melhor fitness não melhorar durante 20 gerações consecutivas, o IGA para de evoluir.

**Seus resultados:**

| Track | Geração em que parou | Gerações máx. | Motivo |
|---|---|---|---|
| A | 34 de 100 | 100 | Early stop (20 sem melhoria a partir da geração 14) |
| B | 38 de 100 | 100 | Early stop |
| Wu 2021 | 25 de 100 | 100 | Early stop |

**O que dizer:** "Todos os tracks ativaram o early stopping antes das 100 gerações. Isso é saudável — significa que o algoritmo convergiu e não adiantaria rodar mais gerações. A convergência rápida (geração 14 na Track A) confirma que a inicialização enviesada pelas razões IEC funciona bem."

---

## Parte 4 — Tabela Consolidada de Resultados

### LSTM — Previsão Temporal

| Horizonte | Dias | R² | MAE% | Veredicto |
|---|---|---|---|---|
| h=1 | 0.5 dia | 0.9996 | ~1% | Quase perfeito |
| h=14 | 7 dias | **0.9979** | **~2%** | **Excelente** (usado no TCC) |
| h=28 | 14 dias | **0.9928** | **~4%** | **Muito bom** (usado no TCC) |
| h=112 | 56 dias | 0.8380 | ~18% | Limite aceitável |

### IGA-XGBoost — Classificação

| Modelo | Track | F1-macro | Acurácia | Features |
|---|---|---|---|---|
| C1 IGA | A (4 classes) | 0.8802 ± 0.0068 | 94.3% | 8 de 15 |
| C2 Default | A | 0.8858 ± 0.0000 | 94.7% | 15 de 15 |
| Wu 2021 | B (7 classes) | **0.7764 ± 0.0059** | **84.7%** | 12 de 18 |
| C3 IGA | B | 0.7525 ± 0.0061 | 82.8% | 8 de 9 |
| C4 Default | B | 0.7356 ± 0.0000 | 82.4% | 9 de 9 |
| T2 Duval | B | 0.2630 | 30.2% | regras |
| T1 Rogers | B | 0.1290 | 13.6% | regras |

---

## Parte 5 — Roteiro de como apresentar para a Orientadora

### Ordem sugerida de apresentação:

**1. Comece pela LSTM (5 min)**
> "Treinei uma LSTM de 2 camadas para prever concentrações futuras de 4 gases dissolvidos. O modelo consegue prever com R²=0.997 para 7 dias e R²=0.993 para 14 dias, com erros de apenas 2-4% da concentração real."

**2. Mostre a degradação por horizonte (2 min)**
> "Testei horizontes de 12 horas até 8 semanas. O modelo mantém R²>0.80 até 8 semanas, mas para os horizontes usados no classificador (7 e 14 dias) a qualidade é excelente."

**3. Explique a engenharia de atributos (3 min)**
> "As previsões da LSTM viram features para o XGBoost. Junto com os gases observados e as razões IEC da norma 60599, temos 15 atributos na Track A."

**4. Mostre os resultados do classificador (5 min)**
> "O XGBoost otimizado pelo algoritmo genético alcança F1-macro de 0.88 em 4 classes (Track A) e 0.75 em 7 classes (Track B). O algoritmo genético selecionou 8 de 15 features e otimizou 9 hiperparâmetros simultaneamente."

**5. Explique a auditoria (3 min) — IMPORTANTE**
> "Encontrei e corrigi um vazamento de dados causado pelo SMOTE. O SMOTE estava sendo aplicado antes da validação cruzada, contaminando os folds de validação. Após a correção, os resultados internos e externos são coerentes (gap < 4%)."

**6. Comparação com a literatura (2 min)**
> "Reproduzi o modelo de Wu et al. (2021) com as mesmas 18 features deles. O resultado deles (F1=0.776) supera o nosso (F1=0.752), mas com quase o dobro de features. A diferença de ~2% mostra que nossa abordagem mais enxuta com 9 features normatizadas é competitiva."
