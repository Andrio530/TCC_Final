# Roteiro Completo de Apresentação — Defesa de TCC (20 Minutos)

**Título:** *PREDIÇÃO DINÂMICA DE FALHAS EM TRANSFORMADORES: Abordagem Híbrida de Previsão Temporal (LSTM) e Classificação Otimizada (IGA-XGBoost)*  
**Autor:** Ândrio Gabriel Epping  
**Orientadora:** Profa. Me. Janaína Conceição Sutil Lemos  
**Instituição:** Universidade do Vale do Rio dos Sinos (UNISINOS) — Curso de Engenharia da Computação (2026)  
**Arquivo da Apresentação:** [defesa_tcc.pptx](file:///c:/Users/ResTIC55/Documents/TCC_Final/defesa_tcc.pptx) (23 slides: 15 principais cronometrados + 1 encerramento + 1 divisor + 6 slides de apoio técnico)

---

## Quadro-Resumo de Controle de Tempo (Meta: 20:00)

| Slide | Seção da Defesa | Duração | Tempo Acumulado | Apoio Visual / Ênfase |
| :--- | :--- | :---: | :---: | :--- |
| **01** | Capa Institucional | 0:20 | **0:20** | Identificação formal e título |
| **02** | Contextualização: Transformadores e DGA | 1:20 | **1:40** | Diagrama da cadeia causal gás-falha |
| **03** | Problema de Pesquisa e Motivação | 1:20 | **3:00** | Três lacunas e pergunta científica central |
| **04** | Objetivos e Síntese das Contribuições | 1:00 | **4:00** | Separação entre proposto vs. demonstrado |
| **05** | Arquitetura Geral Implementada | 1:30 | **5:30** | Fluxograma real dos Tracks A e B |
| **06** | Dados e Blindagem contra Vazamento | 1:30 | **7:00** | Auditoria (2.321 → 1.962) e 4 barreiras anti-leakage |
| **07** | Módulo Temporal: Stacked LSTM (Track A) | 2:00 | **9:00** | Arquitetura (50×4) + extrapolação recursiva |
| **08** | Classificação Evolutiva: IGA-XGBoost | 2:00 | **11:00** | Cromossomo bipartido + mlogloss vs. F1-macro |
| **09** | Desenho Experimental (Track A vs. Track B) | 2:00 | **13:00** | Ablação temporal + recorte justo de 6 falhas |
| **10** | Métricas e Rigor Estatístico | 1:00 | **14:00** | CV 5×5 vs. 5 seeds vs. Bootstrap pareado |
| **11** | Resultados LSTM (h=1, Multi-Horizonte e Persistência) | 1:30 | **15:30** | R²=0,9996 vs. inércia temporal (R²_pers=0,9970) |
| **12** | Resultados de Classificação (Tracks A e B) | 1:30 | **17:00** | Tabelas completas + Trade-off Complexidade vs. F1 |
| **13** | Bootstrap Pareado (5.000×) e TreeSHAP | 1:30 | **18:30** | Área A (p=0,238; p=0,0002) + Área B (23% C₂H₄ e H₂) |
| **14** | Discussão Crítica e Limitações | 1:00 | **19:30** | Quadrante de evidências, teto empírico e limites |
| **15** | Conclusões e Trabalhos Futuros | 0:30 | **20:00** | Resposta direta à pergunta + RUL e Δppm/Δt |
| **16** | Encerramento Institucional | — | — | Agradecimento antes da arguição |
| **17–23** | Material de Apoio (A1 a A6) | Arguição | — | Slides sob demanda para perguntas da banca |

> **Dica de ritmo:** O primeiro bloco (Slides 1–4) encerra exatamente em **4:00**. A metodologia e modelagem (Slides 5–10) termina em **14:00**, garantindo **6 minutos inteiros** para resultados, validação estatística, XAI e conclusões (Slides 11–15). Caso perceba atraso de 30s após o Slide 9, encurte o detalhamento das sementes no Slide 10 (indo direto ao conceito do bootstrap pareado).

---

## SLIDE 1 — CAPA (0:20 | Acumulado: 0:20)

- **Objetivo:** Apresentar formalmente o trabalho, a autoria, a orientação e o vínculo institucional sem alongar a introdução.
- **Mensagem Central:** Proposta de investigação híbrida entre previsão temporal e classificação evolutiva explicável aplicada ao diagnóstico de transformadores por Análise de Gases Dissolvidos (DGA).
- **Fala Sugerida (~45 palavras):**
  > "Bom dia [ou boa noite] aos membros da banca examinadora e à professora orientadora Janaína Lemos. Sou Ândrio Gabriel Epping, graduando em Engenharia da Computação pela UNISINOS, e apresento meu Trabalho de Conclusão de Curso intitulado: *Predição Dinâmica de Falhas em Transformadores: Abordagem Híbrida de Previsão Temporal (LSTM) e Classificação Otimizada (IGA-XGBoost)*."
- **Frase de Transição:**
  > "Para situar o problema, começo explicando o papel físico dos gases dissolvidos no diagnóstico desses equipamentos."
- **Possíveis Perguntas da Banca:** Não aplicável neste slide.

---

## SLIDE 2 — CONTEXTUALIZAÇÃO: TRANSFORMADORES E DGA (1:20 | Acumulado: 1:40)

- **Objetivo:** Fundamentar a relevância estratégica dos transformadores de potência e explicar o vínculo termodinâmico entre falha elétrica/térmica e assinatura gasosa no óleo isolante.
- **Mensagem Central:** Estresses térmicos e elétricos degradam o sistema óleo-papel gerando assinaturas específicas de hidrogênio e hidrocarbonetos que permitem diagnóstico em operação contínua.
- **Fala Sugerida (~175 palavras):**
  > "Transformadores de potência são ativos críticos na transmissão e distribuição elétrica. Durante a operação, estresses térmicos, elétricos e envelhecimento natural rompem ligações químicas do óleo mineral isolante e do papel Kraft celulósico. Os produtos dessa degradação permanecem dissolvidos no óleo, razão pela qual a Análise de Gases Dissolvidos — ou DGA — atua como um exame preventivo do equipamento.
  >
  > Como ilustra a cadeia causal à esquerda, diferentes regimes de energia geram combinações características de gases. Descargas parciais de baixa energia produzem predominantemente hidrogênio (H₂) e metano (CH₄). Sobreaquecimentos evoluem conforme a faixa térmica: de metano e etano (C₂H₆) abaixo de 300 °C até picos de etileno (C₂H₄) acima de 700 °C. Já arcos elétricos severos acima de 1.000 °C quebram ligações triplas gerando acetileno (C₂H₂).
  >
  > Hoje, além das coletas laboratoriais periódicas, sensores online fornecem leituras contínuas dessas concentrações. Isso abre uma questão natural de engenharia: será que projetar a evolução futura desses gases contribui para antecipar ou aprimorar o diagnóstico em comparação à análise estática?"
- **Frase de Transição:**
  > "Para transformar essas medições físicas em laudos confiáveis, entretanto, a literatura atual esbarra em limitações práticas concretas."
- **Possíveis Perguntas da Banca:**
  - *P: Qual é a origem química distinta de CO/CO₂ frente aos hidrocarbonetos?*
    - **R:** CO e CO₂ decorrem principalmente da degradação da celulose sólida (papel Kraft), enquanto H₂, CH₄, C₂H₆, C₂H₄ e C₂H₂ originam-se majoritariamente da pirólise e quebra dielétrica do óleo mineral.

---

## SLIDE 3 — PROBLEMA DE PESQUISA E MOTIVAÇÃO (1:20 | Acumulado: 3:00)

- **Objetivo:** Delimitar com precisão as três lacunas (normas clássicas, ML estático não auditado e LSTM desacoplada) e formular a pergunta científica sem afirmar superioridade a priori.
- **Mensagem Central:** Normas clássicas exigem falha prévia e falham em zonas limítrofes; estudos de IA sofrem de métricas infladas sem deduplicação ou falham em testar a utilidade real da previsão LSTM sobre o classificador.
- **Fala Sugerida (~175 palavras):**
  > "Identificamos três lacunas objetivas que motivam esta investigação. Primeiro, as normas clássicas do setor elétrico — o Triângulo de Duval e as razões da IEC 60599 — adotam fronteiras rígidas e partem da premissa de defeito preexistente: elas sequer definem categoria Normal, rotulando transformadores sadios como defeituosos ou gerando laudos Não Determinados.
  >
  > Segundo, a literatura recente de aprendizado de máquina estático frequentemente reporta acurácias superiores a 99%, mas apoia-se em amostras de teste minúsculas — como os 25 exemplos em Wu et al. (2021) — ou utiliza bases compiladas sem remover duplicatas, vazando exemplos idênticos entre treino e teste, além de raramente reportar intervalos de confiança estatísticos ou explicabilidade física.
  >
  > Terceiro, trabalhos baseados em redes LSTM — como Wang et al. (2023) — avaliam o erro da previsão temporal isoladamente, sem executar um teste de ablação que comprove se prever o futuro realmente ajuda o classificador.
  >
  > Assim, o trabalho responde diretamente à pergunta central destacada na caixa inferior: *em bases reais auditadas, acoplar previsões LSTM e seleção evolutiva via IGA ao XGBoost agrega ganho estatisticamente comprovado em relação a modelos estáticos e normas clássicas, preservando coerência físico-química auditável por TreeSHAP?*"
- **Frase de Transição:**
  > "Para responder a essa pergunta sem viés experimental, estruturei os objetivos e separei claramente o que foi proposto do que os experimentos demonstraram."
- **Possíveis Perguntas da Banca:**
  - *P: Por que o Triângulo de Duval gera tantos erros quando aplicado em uma base completa com amostras normais?*
    - **R:** O Triângulo projeta qualquer concentração em coordenadas percentuais (%CH₄, %C₂H₄, %C₂H₂) que somam 100%, dividindo o triângulo apenas entre zonas de falha (PD, D1, D2, T1, T2, T3 e DT); logo, toda amostra Normal com pequenas frações residuais acaba caindo em alguma região de falha (predominantemente T1 neste banco, com 89,0%).

---

## SLIDE 4 — OBJETIVO E CONTRIBUIÇÕES (1:00 | Acumulado: 4:00)

- **Objetivo:** Apresentar o objetivo geral fiel ao texto do TCC, os 5 objetivos específicos e a matriz transparente entre o proposto e o comprovado.
- **Mensagem Central:** O rigor do trabalho não reside em "forçar" um resultado positivo para a arquitetura híbrida, mas em isolar e comprovar cada contribuição, inclusive o resultado negativo da redundância temporal.
- **Fala Sugerida (~135 palavras):**
  > "Nosso objetivo geral, fiel ao documento, é desenvolver e avaliar uma metodologia computacional baseada em DGA para investigar o valor efetivo de previsões temporais por LSTM, a simplificação de variáveis pelo IGA-XGBoost e a coerência físico-química frente à IEC 60599 e ao Triângulo de Duval.
  >
  > Os cinco objetivos específicos desdobram esse caminho: auditar e estruturar os eixos Track A e Track B; treinar a LSTM multi-horizonte; testar a hipótese híbrida com bootstrap pareado; otimizar atributos e hiperparâmetros via IGA; e auditar decisões com TreeSHAP.
  >
  > A tabela à direita separa honestamente implementação de comprovação: implementamos a arquitetura híbrida completa, mas demonstramos que a LSTM — apesar de atingir R² acima de 0,99 — é estatisticamente redundante para diagnosticar o estado pontual atual (p = 0,238). Por outro lado, comprovamos redução de 53% dos atributos no Track A, superação robusta das normas no Track B e recuperação autônoma do corte físico de 23% de etileno via TreeSHAP."
- **Frase de Transição:**
  > "Vejamos agora como essa proposta foi traduzida exatamente na arquitetura de software em dois eixos."
- **Possíveis Perguntas da Banca:**
  - *P: Por que o TCC não utilizou uma única base com séries temporais E as 7 classes normativas?*
    - **R:** Porque nenhuma base pública disponível reúne simultaneamente séries temporais longas por equipamento e rotulagem padronizada nas 7 categorias IEC sobre os 5 gases completos (Katser tem 4 gases sem CH₄/C₂H₆ e 4 classes; a base compilada tem os 5 gases e 7 classes, mas é estática).

---

## SLIDE 5 — ARQUITETURA GERAL PROPOSTA (1:30 | Acumulado: 5:30)

- **Objetivo:** Reconstruir a arquitetura real implementada no código-fonte, distinguindo os fluxos principais de Track A e Track B e os baselines de comparação.
- **Mensagem Central:** Track A testa a hipótese temporal com ablação rigorosa (C1, C2, C5a, C5b, C6); Track B testa o benchmark multiclasse IEC completo com CustomSMOTE encapsulado na CV (C3, C4, M3, M2, M1).
- **Fala Sugerida (~190 palavras):**
  > "Este diagrama reconstrói o fluxo real executado pelos scripts do repositório, sem abstrações genéricas. O código organiza dois pipelines independentes.
  >
  > Na faixa superior está o **Track A**, dedicado à hipótese temporal sobre 3.000 séries de Katser com 4 gases. O `ScalerPerGas`, ajustado apenas nas 1.785 séries de treino, alimenta uma janela deslizante de 50 passos na rede Stacked LSTM. As projeções de 1 semana (h=14) e 2 semanas (h=28) geram 8 atributos futuros, que se unem aos 4 gases observados em t=420 e a 5 razões adaptadas — compondo 17 candidatos balanceados via `sample_weight` para o IGA-XGBoost (C1). Logo abaixo, mantemos os fluxos alternativos de ablação sobre o mesmo conjunto de teste: C2 (referência fixa), C5a e C5b (removendo a LSTM na origem) e C6 (usando apenas as previsões).
  >
  > Na faixa inferior está o **Track B**, voltado ao diagnóstico completo nas 7 classes da IEC 60599. Os 2.321 registros brutos passam por auditoria de duplicatas resultando em 1.962 amostras únicas de 5 gases. Calculamos 14 indicadores físico-químicos em ppm antes da escala logarítmica e Min-Max. Dentro da validação cruzada do IGA executa um `ImbPipeline` com o `CustomSMOTE` termodinâmico, produzindo C3 (16 atributos selecionados) confrontado com C4, Wu et al. (M3), linha base bruta, Duval (M2) e IEC (M1). À direita, ambos os eixos convergem para bootstrap pareado e TreeSHAP."
- **Frase de Transição:**
  > "O ponto de partida crítico dessa arquitetura é a integridade dos dados; passo agora às decisões anti-vazamento do pré-processamento."
- **Possíveis Perguntas da Banca:**
  - *P: Qual a diferença operacional entre os dois pipelines no módulo Python?*
    - **R:** Track A usa `src/lstm/trainer.py` + `src/iga_xgboost/train_hybrid.py` com pesos analíticos `sample_weight`; Track B usa `src/iga_xgboost/train_classification.py` encapsulando `CustomSMOTE` em `imblearn.Pipeline`, pois no Track A não sintetizamos amostras temporais via SMOTE.

---

## SLIDE 6 — DATASET E PRÉ-PROCESSAMENTO (1:30 | Acumulado: 7:00)

- **Objetivo:** Apresentar números auditados de registros, divisões, classes, espaço de atributos e as quatro barreiras explícitas contra data leakage.
- **Mensagem Central:** A remoção de 359 registros inválidos/duplicados (15,5%) no Track B e o encapsulamento estrito de scalers e `CustomSMOTE` no treino evitam o otimismo artificial comum à literatura.
- **Fala Sugerida (~190 palavras):**
  > "A tabela à esquerda consolida as especificações verificáveis de cada eixo. No **Track A**, são 3.000 séries de 420 leituras a cada 12 horas em escala normalizada (~10^-3), com 0 duplicatas exatas por hash MD5. Dividimos em 1.785 séries de treino, 315 de validação e 900 de teste oficial — onde 81,2% pertencem à classe Normal e todas as séries mantêm rótulo constante ao longo dos 210 dias.
  >
  > No **Track B**, auditamos a base compilada de 2.321 registros em ppm dos 5 gases normativos. Removemos 303 réplicas exatas e 56 registros com mesma concentração gasosa e rótulos conflitantes — eliminando 15,5% da base original e restando 1.962 amostras únicas, particionadas de forma estratificada em 70/15/15 (1.372 treino, 295 validação, 295 teste).
  >
  > Destaco à direita quatro decisões metodológicas críticas contra vazamento: primeiro, toda razão molar e proporção de Duval (TH, TH₁) é calculada em ppm bruto **antes** da compressão `ln(x+1)` e Min-Max, preservando o significado físico; segundo, escalonadores são ajustados exclusivamente no treino; terceiro, as janelas w=50 da LSTM jamais cruzam fronteiras entre transformadores; e quarto, no Track B implementamos um `CustomSMOTE` termodinâmico dentro da validação cruzada."
- **Frase de Transição:**
  > "Com os dados estruturados e protegidos, detalho o primeiro estágio do modelo: a previsão temporal com Stacked LSTM."
- **Possíveis Perguntas da Banca:**
  - *P: Como funciona o CustomSMOTE termodinâmico passo a passo?*
    - **R:** A classe `CustomSMOTE` (em `src/iga_xgboost/train_classification.py`) isola exclusivamente os 5 gases brutos escalonados para gerar os vizinhos sintéticos via SMOTE (`k=5`), inverte Min-Max e `expm1` para retornar a ppm físico, recalcula exatamente as 14 razões, totais TH/TH₁ e frações percentuais a partir dos 5 gases sintéticos e reaplica o escalonamento — impedindo combinações lineares fisicamente impossíveis nas razões.

---

## SLIDE 7 — MODELO LSTM: PREVISÃO TEMPORAL (2:00 | Acumulado: 9:00)

- **Objetivo:** Demonstrar tecnicamente a arquitetura exata implementada em `src/lstm/model.py` e `trainer.py`, as dimensões dos tensores, o papel da camada densa e o mecanismo de extrapolação recursiva.
- **Mensagem Central:** Uma Stacked LSTM com duas camadas de 64 unidades projeta o estado oculto final de 64 dimensões para 4 concentrações via camada Densa linear; horizontes de 1 e 2 semanas são gerados recursivamente.
- **Fala Sugerida (~250 palavras):**
  > "No estágio de modelagem temporal, avaliamos comprimentos de janela candidatos de 10, 20, 30 e 50 passos sobre as 315 séries de validação. A janela **w*=50** — correspondente a 25 dias de histórico a cada 12 horas — alcançou o menor MAE e definiu a entrada `(50, 4)` para os 4 gases do Track A.
  >
  > Como mostra o diagrama fiel a `src/lstm/model.py`, a rede é uma **Stacked LSTM** com duas camadas recorrentes de 64 células cada. A primeira camada retorna toda a sequência de estados ocultos `(B, 50, 64)` mantendo o contexto temporal passo a passo para a segunda camada recorrente; entre elas atua um `Dropout(0,20)` para regularização. A segunda camada consolida a dinâmica da janela no último vetor de estado oculto **h_50 em R^64**. Após um segundo `Dropout(0,20)`, uma camada **Dense com 4 neurônios e ativação linear** transforma a representação latente de 64 dimensões nas quatro concentrações previstas em `t+1`. Optou-se pela ativação linear para não saturar previsões contínuas caso valores de teste ultrapassem ligeiramente `[0, 1]`.
  >
  > O treinamento (tabela inferior direita) empregou MSE como função de perda, otimizador Adam (`lr=10^-3`, `batch_size=1.024`) sobre 660.450 janelas de treino, `ReduceLROnPlateau` e `EarlyStopping` (paciência 15, convergindo em média por volta da época 16) em cinco sementes. Por fim, para obter previsões em `h=14` (1 semana) e `h=28` (2 semanas), realizamos extrapolação recursiva: a previsão `ŷ_{t+1}` entra no final da janela FIFO descartando o ponto mais antigo, repetindo iterativamente sem usar dados reais futuros."
- **Frase de Transição:**
  > "Essas projeções temporais entram no estágio seguinte: a seleção evolutiva de atributos e classificação com IGA-XGBoost."
- **Possíveis Perguntas da Banca:**
  - *P: Por que usar extrapolação recursiva em vez de saída direta Multi-Output de 56 dimensões (`14 × 4` + `28 × 4`)?*
    - **R:** Uma única saída autoregressiva `h=1` permite inspecionar a degradação contínua do R² ao longo de 12 horizontes até 8 semanas (112 passos) com um único modelo compacto e identificar exatamente onde o limite de R² >= 0,80 se rompe, evitando explosão paramétrica na camada final.

---

## SLIDE 8 — IGA E XGBOOST: CLASSIFICAÇÃO (2:00 | Acumulado: 11:00)

- **Objetivo:** Explicar representação do cromossomo bipartido, ciclo genético, hiperparâmetros otimizados e a distinção rigorosa entre função objetivo interna (`mlogloss`) e métrica de fitness/avaliação (`F1-macro`).
- **Mensagem Central:** O IGA acopla uma máscara binária com viés inicial IEC e 9 genes contínuos de hiperparâmetros avaliados por F1-macro em CV 5-fold, enquanto o XGBoost otimiza `mlogloss` regularizada em cada árvore.
- **Fala Sugerida (~250 palavras):**
  > "Na etapa de classificação, implementamos o **IGA-XGBoost** (`src/iga_xgboost/cromossomo.py` e `iga.py`) para resolver simultaneamente dois problemas acoplados: **quais atributos reter** e **como parametrizar o gradiente boosting**.
  >
  > No topo, vemos a codificação híbrida de cada indivíduo: os primeiros *p* genes formam uma máscara binária — 17 bits no Track A ou 19 no Track B, com mínimo de 2 atributos ativos — onde introduzimos um viés de inicialização de domínio ativando razões IEC com probabilidade 0,80 contra 0,50 para as demais. Os 9 genes finais são reais em `[0, 1]` decodificados afimamente para os hiperparâmetros críticos do XGBoost: taxa de aprendizado, profundidade, número de árvores, subamostragem e termos de regularização.
  >
  > O fluxo evolutivo parte de 50 indivíduos, avaliados estritamente sobre o conjunto de treino pela média do **F1-macro em CV estratificada de 5 dobras** (sem tocar na validação ou no teste). Aplicamos torneio de tamanho 3, elitismo dos 2 melhores, crossover uniforme na máscara e BLX-alfa (`0,50`) nos genes contínuos (`P_c=0,90`), além de mutação bit-flip e gaussiana (`P_m=0,10`), com parada em 100 gerações ou 20 sem melhoria.
  >
  > Destaco à direita uma distinção conceitual essencial: na construção interna das árvores, o XGBoost minimiza a perda **`mlogloss`** com regularização estrutural de 2ª ordem (`gamma`, `lambda`, `alpha`); já o fitness do IGA e a avaliação final utilizam **F1-macro**. Reparem na tabela como no Track B o IGA elevou `max_depth` para 12, mas compensou com `colsample_bytree = 0,3433` e `reg_lambda = 3,9425` para conter sobreajuste."
- **Frase de Transição:**
  > "Vejamos agora como essa otimização se insere na matriz experimental completa dos Tracks A e B."
- **Possíveis Perguntas da Banca:**
  - *P: O IGA foi implementado em Python puro (`iga.py`) sem bibliotecas genéticas externas; como garantiu a não-degeneração de indivíduos?*
    - **R:** O módulo `cromossomo.py` força o mínimo de 2 atributos ativos atribuindo fitness 0,0 a cromossomos degenerados (<2 bits), usa `clip` `[0,1]` pós-BLX-alfa/mutação e mantém elitismo dos 2 melhores garantindo monotonicidade da população.

---

## SLIDE 9 — PROTOCOLOS EXPERIMENTAIS: TRACK A vs. TRACK B (2:00 | Acumulado: 13:00)

- **Objetivo:** Consolidar o desenho experimental dos dois eixos, todas as configurações de ablação (C1–C6) e baselines externos (M1–M3 e Linha Base Bruta), explicando a impossibilidade e o propósito da separação.
- **Mensagem Central:** Track A isola a utilidade causal das projeções LSTM com 5 controles pareados; Track B compara o IGA-XGBoost ao estado da arte (Wu et al.) e às normas em 7 classes e no recorte justo de 6 falhas ativas.
- **Fala Sugerida (~240 palavras):**
  > "Para responder separadamente às perguntas do TCC sem misturar domínios incompatíveis, estruturamos os dois protocolos experimentais detalhados nesta matriz.
  >
  > O **Track A** isola a pergunta temporal sobre as 900 séries de teste em 4 classes. Avaliamos cinco configurações principais: **C2** é a referência fixa com os 17 atributos incluindo LSTM; **C1** aplica o IGA sobre esses 17; **C5a** e **C5b** representam os controles estáticos puros — usando apenas os 9 atributos observados em `t=420`, sem LSTM desde a origem; e **C6** utiliza exclusivamente as 8 projeções futuras. Também registramos a ablação pós-hoc sobre C1, explicando por que C5a/C5b é o contraste justo.
  >
  > O **Track B** conduz o benchmark multiclasse normativo com 5 gases: nosso **C3** (IGA-XGBoost, selecionando 16 de 19 variáveis), **C4** (XGBoost fixo nos 19 atributos), **M3** (reimplementação do conjunto de 18 atributos de Wu et al., 2021 com nosso IGA), a **Linha Base Bruta** com apenas 5 gases em ppm, o **Triângulo 1 de Duval (M2)** e as **Razões IEC 60599 (M1)**. Como normas clássicas não contemplam a classe Normal, avaliamos todo o Track B tanto nas 7 classes globais (`N=295`) quanto no recorte justo de **6 falhas ativas (`N=189`)**, além da validação cruzada repetida 5×5 sobre as 1.667 amostras de desenvolvimento."
- **Frase de Transição:**
  > "Antes dos números, esclareço rapidamente quais métricas governam cada etapa e por que não confundimos variância entre sementes com significância estatística."
- **Possíveis Perguntas da Banca:**
  - *P: Por que C1 pós-hoc (remover LSTM de C1 sem re-otimizar) não serve para provar ganho da LSTM?*
    - **R:** Porque durante a busca genética do C1, o IGA substituiu gases observados altamente informativos (`H₂`, `C₂H₄`, `C₂H₂`) pelas projeções LSTM equivalentes; remover essas colunas a posteriori deixa apenas 4 atributos mutilados, enquanto C5a/C5b refaz o modelo estático completo sem LSTM.

---

## SLIDE 10 — MÉTRICAS E VALIDAÇÃO ESTATÍSTICA (1:00 | Acumulado: 14:00)

- **Objetivo:** Distinguir otimização, seleção e avaliação final; justificar F1-macro; e diferenciar variância de 5 sementes, CV 5×5 e bootstrap pareado estratificado (5.000×).
- **Mensagem Central:** F1-macro protege as classes minoritárias raras, e apenas o IC de 95% do bootstrap pareado estratificado nos mesmos índices de teste autoriza afirmar diferença estatística entre modelos.
- **Fala Sugerida (~140 palavras):**
  > "Este slide resume os critérios estatísticos adotados. No treinamento, LSTM minimiza MSE e XGBoost minimiza `mlogloss`. Na seleção interna, `w*` é escolhido pelo menor MAE na validação e o IGA maximiza F1-macro médio em CV 5-fold no treino — adicionando no Track B uma CV repetida 5×5 (25 dobras) para verificar estabilidade fora da partição única.
  >
  > No teste final, adotamos **F1-macro** como métrica primária por atribuir peso igual às classes: no Track A, 81,2% das amostras são normais, de modo que acurácia global alta mascararia falhas críticas como PD e T1. Por fim, separamos três fontes de incerteza: a média em 5 sementes mede estabilidade estocástica de treino; a CV 5×5 mede variabilidade de partição; e o **bootstrap pareado estratificado (5.000 réplicas)** reamostra os modelos sobre os mesmos índices de teste. Afirmamos ganho significativo somente quando o IC 95% de Delta F1 exclui o zero."
- **Frase de Transição:**
  > "Com as regras de validação estabelecidas, passemos aos resultados experimentais, iniciando pelo Estágio 1 — a previsão temporal com LSTM."
- **Possíveis Perguntas da Banca:**
  - *P: Por que usar bootstrap pareado estratificado no teste em vez de um teste t sobre as 5 sementes?*
    - **R:** Porque rodar 5 sementes testa apenas variação aleatória interna do treinamento sobre um mesmo conjunto fixo de teste (o que viola independência estatística como estimativa de erro amostral); já o bootstrap estratificado por classe reamostra as 900 ou 295 observações reais preservando o suporte pequeno de classes como T1/T2 (N=16) e pareando diretamente os dois modelos por amostra.

---

## SLIDE 11 — RESULTADOS DA PREVISÃO TEMPORAL (LSTM) (1:30 | Acumulado: 15:30)

- **Objetivo:** Reportar com fidelidade as métricas da LSTM em `h=1` e multi-horizonte, confrontar com o baseline de persistência e apresentar a interpretação física da inércia temporal.
- **Mensagem Central:** A LSTM atinge R² = 0,9996 em 12h e mantém R² >= 0,80 até 7 semanas (`h=98`), mas o preditor ingênuo de persistência já atinge R² = 0,9970 em 1 semana devido à alta inércia dos dados.
- **Fala Sugerida (~190 palavras):**
  > "Na Tabela 1 superior, sobre as 333.000 janelas do conjunto de teste cego de 900 transformadores, a Stacked LSTM atingiu para 1 passo à frente (`h=1`, 12 horas) **MAE médio de 5,59×10^-5** na escala original da base e **R² de 0,9996 ± 0,0001** em cinco sementes, mantendo consistência em todos os quatro gases.
  >
  > Na análise recursiva multi-horizonte (Tabela 2 e gráfico superior direito), a rede sustenta **R² > 0,99 e MAE% abaixo de 4%** até 2 semanas (`h=14` e `h=28`, exatamente os dois horizontes acoplados ao XGBoost). O critério operacional uniforme de **R² >= 0,80** em todos os gases é satisfeito até 7 semanas (`h=98`, com H₂ em 0,849), rompendo-se apenas na 8ª semana (`h=112`), quando o hidrogênio cai para 0,788.
  >
  > Contudo, o diagnóstico mais revelador está na segunda coluna da Tabela 2: comparando a LSTM contra o **baseline de persistência** (`ŷ_{t+h} = y_t`), repetir o valor presente já alcança R² de **0,9970 em 1 semana** e **0,9886 em 2 semanas**. Embora as trajetórias previstas à direita sigam com fidelidade o comportamento real, os gases evoluem tão gradualmente sob operação monitorada que as projeções são quase colineares ao valor atual."
- **Frase de Transição:**
  > "Vejamos exatamente qual foi a consequência dessa colinearidade temporal ao acoplar essas previsões ao classificador XGBoost."
- **Possíveis Perguntas da Banca:**
  - *P: Por que o MAE absoluto de CO (16,15×10^-5) é cerca de 85 vezes maior que o de C2H2 (0,19×10^-5)?*
    - **R:** Pela diferença natural de escala e variância entre os gases na base Katser original: o CO apresenta maior magnitude de variação absoluta, mas em métricas adimensionais (`R² = 0,9997` vs. `0,9996` e `MAE% = 0,88%` vs. `1,00%` em `h=1`), a qualidade relativa é virtualmente idêntica.

---

## SLIDE 12 — RESULTADOS COMPARATIVOS DE CLASSIFICAÇÃO (1:30 | Acumulado: 17:00)

- **Objetivo:** Apresentar integralmente os números das ablações Track A e dos benchmarks Track B, interpretando redução dimensional, equivalência temporal e teto empírico.
- **Mensagem Central:** No Track A, o IGA reduz de 17 para 8 atributos sem perda, mas C5a estático iguala o escore; no Track B, C3/C4/M3 superam drasticamente as normas mas convergem para ~0,76–0,80.
- **Fala Sugerida (~190 palavras):**
  > "Este slide apresenta os dois quadros de classificação sem jamais misturar as métricas entre eixos. No **Track A** (tabela superior), o modelo completo fixo C2 com 17 atributos obtém **F1-macro de 0,8911** e o **C1 otimizado pelo IGA** atinge **0,8909** — cortando **53% dos atributos (17 para 8)** sem qualquer perda preditiva. Entretanto, ao avaliarmos **C5a**, treinado exclusivamente com os 9 atributos estáticos presentes sem LSTM, o escore é **0,8855** — com total sobreposição dos intervalos de confiança em relação a C1 e C2. Note-se que a queda da linha pós-hoc (`0,8280`) é apenas artefato por retirar variáveis escolhidas pelo IGA sem reotimizar; a ablação justa é C5a e C5b.
  >
  > No **Track B** (tabela inferior), nosso **C3 (IGA-XGBoost)** alcança **F1 = 0,7970** no teste global de 7 classes e **0,8000** nas 6 falhas ativas (`N=189`), superando largamente o Triângulo de Duval (`0,4499`) e a IEC 60599 (`0,3301`). Contudo, C4 (`0,7843`) e Wu et al. M3 (`0,7829`) empatam dentro da margem estatística e estabilizam juntos entre **0,752 e 0,765 na CV 5×5**, claramente acima da linha base com 5 gases brutos (`0,7082` CV). O gráfico à direita resume essa fronteira de complexidade versus F1."
- **Frase de Transição:**
  > "Para validar formalmente quais diferenças são significativas e explicar a física interna desses modelos, passo aos testes pareados e ao TreeSHAP."
- **Possíveis Perguntas da Banca:**
  - *P: A diferença entre F1=0,7970 de C3 no teste e 0,7516 na CV 5×5 indica overfitting no teste?*
    - **R:** Não: na CV 5×5 avaliamos a estabilidade paramétrica sem o `CustomSMOTE` interno, além do tamanho amostral moderado do teste (`N=295`, com apenas 16 exemplos em T1 e T2 — na validação única foi `0,7038`). O IC 95% `[0,739; 0,852]` engloba naturalmente a média da CV 5×5.

---

## SLIDE 13 — BOOTSTRAP PAREADO E TREESHAP (1:30 | Acumulado: 18:30)

- **Objetivo:** Consolidar na mesma visão a prova inferencial (bootstrap 5.000×) e a explicabilidade físico-química (TreeSHAP), respeitando correlação ≠ causalidade.
- **Mensagem Central:** Bootstrap pareado comprova redundância da LSTM (`p=0,238`) e equivalência C3/C4/M3, ao passo que comprova ganho real sobre normas (`p=0,0002`); TreeSHAP explica fisicamente a migração de crédito e o corte de 23% de etileno.
- **Fala Sugerida (~190 palavras):**
  > "Na **Área A à esquerda**, o bootstrap estratificado pareado (5.000 réplicas) encerra qualquer ambiguidade inferencial. No Track A, o contraste **C2 versus C5a** gera Delta F1 de `+0,0056` com IC 95% de `[-0,0037; +0,0150]` (`p=0,2384`): como o intervalo contém zero, não há diferença estatística ao incluir a LSTM. O mesmo vale entre **C1 e C2** (`p=0,9956`) e, no Track B, entre **C3, C4 e Wu et al.** (`p=0,182` e `0,468`). Já contra as normas em falhas ativas, o ganho de **+0,3500 sobre Duval** e **+0,4699 sobre IEC** possui **p = 0,0002**.
  >
  > Na **Área B à direita**, o **TreeSHAP em log-odds** — com estabilidade de ranking Spearman **rho = 0,9541** entre 5 sementes — dá sentido físico aos resultados. No Track B (topo direito), o agrupamento por famílias confirma descargas regidas por acetileno e térmicas por etileno/etano; no efeito principal de `ratio_C2H4_TH` para D1, a contribuição cruza zero entre **20% e 25%**, recuperando autonomamente a fronteira canônica de **23% de Duval**! No Track A (abaixo), o gráfico de migração prova que `lstm_h14_H2` e `lstm_h28_H2` trocam crédito entre sementes mantendo soma fixa em `0,4734 ± 0,0131` — diagnóstico visual claro de redundância colinear, recordando que valores SHAP medem atribuição no modelo e não causalidade direta."
- **Frase de Transição:**
  > "Com todas as evidências quantitativas e interpretáveis fechadas, apresento nossa leitura crítica e limitações."
- **Possíveis Perguntas da Banca:**
  - *P: Por que outros limiares SHAP, como `C2H2/TH` (~8%) ou `C2H4/C2H6` (~2,97), diferiram da norma (13% e 4,0)?*
    - **R:** Porque a base industrial compilada possui casos rotulados como D1/D2 com acetileno menor que 13% e principalmente uma anomalia nos casos T3 históricos cuja mediana `C₂H₄/C₂H₆` é `0,25` (confirmada na auditoria normativa do Apoio A5). O XGBoost ajusta seu corte à distribuição empírica real da base.

---

## SLIDE 14 — DISCUSSÃO CRÍTICA (1:00 | Acumulado: 19:30)

- **Objetivo:** Sintetizar com maturidade científica o balanço crítico entre achados positivos, resultados negativos comprovados, teto dos dados e limitações metodológicas.
- **Mensagem Central:** Um trabalho de engenharia maduro delimita exatamente onde a complexidade extra não se paga e expõe honestamente o teto empírico imposto pelas bases públicas.
- **Fala Sugerida (~140 palavras):**
  > "Nossa discussão crítica consolida quatro eixos. Primeiro, o que foi **demonstrado com evidência**: redução dimensional de 53% no Track A preservando F1, superioridade significativa sobre Duval e IEC mesmo em falhas puras (`p=0,0002`) e recuperação de limiares físicos via TreeSHAP (`rho=0,954`).
  >
  > Segundo, o que **não se sustentou**: a hipótese de que projeções LSTM melhoram o diagnóstico pontual sob regime de alta inércia (`r > 0,997`), nem superioridade estatística do IGA sobre bons hiperparâmetros fixos no Track B, onde todos os ensembles param no teto empírico de **0,76 a 0,80** por sobreposição térmico-elétrica e inconsistência de laudos em T3.
  >
  > Terceiro, destacamos duas **implicações**: acurácias acima de 99% na literatura refletem frequentemente duplicatas não removidas ou amostras minúsculas; e na subestação, para diagnóstico do estado atual, o modelo estático C5a economiza toda a complexidade de manter uma LSTM em produção. Quarto, assumimos nossas limitações: fragmentação em dois eixos, projeções de treino do Track A in-sample (com teste 100% cego) e CV 5×5 pós-seleção."
- **Frase de Transição:**
  > "Chegamos finalmente às conclusões diretas e perspectivas de continuidade."
- **Possíveis Perguntas da Banca:**
  - *P: Se as previsões LSTM usadas para treinar o XGBoost do Track A foram geradas sobre séries que treinaram a própria LSTM (sem esquema out-of-fold), isso compromete a conclusão?*
    - **R:** Não na direção do argumento: gerar previsões de treino com a LSTM in-sample tende a **superestimar** a fidelidade das features futuras durante o treinamento do XGBoost (viés otimista para a LSTM); mesmo assim, quando ambos são confrontados nas 900 séries inéditas de teste, a LSTM não vence C5a (`p=0,238`), tornando o resultado negativo ainda mais robusto.

---

## SLIDE 15 — CONCLUSÕES E TRABALHOS FUTUROS (0:30 | Acumulado: 20:00)

- **Objetivo:** Responder diretamente à pergunta de pesquisa, consolidar a principal contribuição/limitação e direcionar trabalhos futuros alinhados aos achados.
- **Mensagem Central:** Projeções LSTM são redundantes para classificar o estado pontual sob evolução gradual; a contribuição do TCC está no protocolo rigoroso anti-vazamento, simplificação pelo IGA, ganho sobre normas e explicabilidade via TreeSHAP, apontando o redirecionamento natural da LSTM para RUL e taxas `Δppm/Δt`.
- **Fala Sugerida (~85 palavras):**
  > "Em resposta direta à nossa pergunta científica: sob evolução lenta de gases dissolvidos, o estado atual já retém o sinal diagnóstico, tornando previsões LSTM estatisticamente redundantes para classificação pontual (`ΔF1=+0,0056`, `p=0,238`). A principal contribuição do TCC é entregar um protocolo reprodutível, auditado contra vazamento, parcimonioso via IGA, superior às normas clássicas e fisicamente validado pelo TreeSHAP — reconhecendo como principal limitação a fragmentação das bases públicas atuais.
  >
  > Como trabalhos futuros, o caminho natural é redirecionar a alta capacidade preditiva da LSTM (`R² > 0,99`) para **prognóstico de Vida Útil Remanescente (RUL)** — prevendo *quando* o transformador cruzará os limites da IEC — além de modelar taxas de variação (`Δppm/Δt`) e integrar carga elétrica e temperatura do óleo. Muito obrigado e coloco-me à disposição da banca."
- **Frase de Transição:**
  > Encerramento (avançar para Slide 16 durante agradecimentos/arguição; acionar Slides 18–23 conforme perguntas).

---

## SLIDES 16 A 23 — ENCERRAMENTO E MATERIAL DE APOIO (ARGUIÇÃO)

- **Slide 16 — Encerramento Institucional:** Tela limpa para aguardar as considerações dos examinadores.
- **Slide 17 — Divisor Material de Apoio:** Separador visual claro caso a banca peça detalhes matemáticos/diagnósticos profundos.
- **Slide 18 (Apoio A1) — Portas LSTM, BPTT, Adam e Curvas de Treino:** Usar se perguntarem sobre vanishing gradient, dimensões `(660.450, 50, 4)` ou por que `val_loss < train_loss` (Dropout 0,20 ativo apenas no treino; nota explícita sobre escala Min-Max vs. escala original Katser).
- **Slide 19 (Apoio A2) — Operadores do IGA, CustomSMOTE e Convergência:** Usar para detalhar BLX-alfa, preservação termodinâmica no `CustomSMOTE` e os exatamente 8 atributos (Track A) e 16 atributos (Track B) escolhidos.
- **Slide 20 (Apoio A3) — Função Objetivo do XGBoost e Controle de Overfitting:** Usar se questionarem `max_depth=12` (explicar efeito combinado de `colsample_bytree=0,3433`, `reg_lambda=3,9425`, `gamma=1,4356` e `min_child_weight=3` dado `h <= 0,25`), com exemplo visual de árvore em D1.
- **Slide 21 (Apoio A4) — Matrizes de Confusão e Métricas por Classe:** Usar se perguntarem onde os modelos mais erram (`Normal<->T1` no Track A; `D1<->D2` e baixo suporte `N=16` em T1/T2 no Track B; explicando o vínculo entre `xgb_confusion_track_b.png` e seed 42 do JSON).
- **Slide 22 (Apoio A5) — Auditoria Normativa (N=1.962) e Triângulo de Duval:** Usar se a banca questionar por que Duval/IEC pontuaram tão baixo — tabela completa provando 89% dos normais em T1 e mediana `C₂H₄/C₂H₆ = 0,25` nos rótulos históricos de T3.
- **Slide 23 (Apoio A6) — SHAP Complementar e Referências:** Usar para mostrar o Heatmap de classes, Beeswarm de D1 e o estudo de caso da Amostra 11 (erro D1→D2 onde a própria geometria de Duval dá razão ao XGBoost), além das referências bibliográficas.
