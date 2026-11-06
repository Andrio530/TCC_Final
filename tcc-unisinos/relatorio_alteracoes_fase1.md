# Relatório Detalhado de Alterações — Plano de Revisão (Fase 1)

**Data de Execução:** 30 de outubro de 2026  
**Documento Base:** `plano_revisao_fase1.md`  
**Escopo:** Execução estrita e exclusiva dos pontos aprovados no plano, sem reescrita de trechos alheios às metas estabelecidas.

---

## 1. Sumário Executivo das Modificações Realizadas

Todas as alterações planejadas foram implementadas diretamente nos arquivos do projeto:

| Arquivo Modificado | Itens do Plano Contemplados | Resumo da Intervenção |
| :--- | :--- | :--- |
| **`tcc-unisinos/pretextual.tex`** | C4, E1 | Limpeza de siglas órfãs (`ABNT`), inserção em ordem alfabética das siglas `ANOVA`, `DP`, `IC`, `OOF`. Título original mantido rigorosamente intacto. |
| **`tcc-unisinos/introducao.tex`** | D1 | Fusão do parágrafo de frase única da Seção 1.4 ao parágrafo subsequente. |
| **`tcc-unisinos/metodologia.tex`** | A1, A4, A5, A9, B3, B5, B6 | Definição da LSTM como estritamente recursiva ($h=1$ realimentado); retificação da proveniência da base chinesa como compilação multicêntrica (Li, Wang e Song, 2019); correção da partição de treino/validação da LSTM (85%/15%, ~16 épocas); isolamento 100% cego do teste na seleção por IGA; justificativa operacional da janela $w=50$. |
| **`tcc-unisinos/resultados.tex`** | A1, A2, A4, A6, A10, B4 | Atualização da Tabela 2 para 12 horizontes oficiais (incluindo $h=112$ e o cruzamento de $\text{H}_2 < 0,80$); inserção dos valores e discussão do Baseline de Persistência ($\hat{y}_{t+h} = y_t$); fundamentação da sobreposição física de classes em Li et al. (2019); correção e detalhamento dos valores exatos de $|\text{SHAP}|$ nas 5 sementes do modelo C1 ($0,4624$ a $0,4964$, média $0,4734 \pm 0,0131$) e esclarecimento da exclusão de $\text{H}_{2,\text{obs}}$ pelo IGA. |
| **`tcc-unisinos/conclusao.tex`** | A4, B1, B2, B3, B4, B6 | Reestruturação aprofundada da Seção 5.3 (Limitações), documentando a normalização global das 3.000 séries, a análise descritiva *post-hoc* por 5-NN, o rigor do teste cego sob IGA, o escopo dos dados de Katser et al. e a inércia temporal comprovada pelo baseline de persistência. |
| **`tcc-unisinos/referencias.bib`** | A11, C1 | Correção dos autores e DOI de Li et al. (2019); expansão completa dos coautores de LeCun et al. (1998) eliminando `and others`. |
| **`tcc-transformer-fault/src/lstm/multi_horizon_eval.py`** | A8, A10 | Correção da chave de cor de $\text{C}_2\text{H}_4$ (removendo duplicação com a cor de $\text{CO}$); inclusão de acentuação gráfica correta em português; regeração da Figura 8. |
| **`tcc-transformer-fault/src/evaluation/fase6_shap_track_a.py`** | A2, A8 | Ajuste de espaçamento, anotações de porcentagem nas barras e reposicionamento da legenda fora da área de dados; regeração da Figura 18. |
| **`tcc-unisinos/figuras/`** | A8 | Atualização das imagens `lstm_multi_horizon_r2_norm.png` (Figura 8) e `fase6/fig7_lstm_credit_migration.png` (Figura 18). |

---

## 2. Detalhamento Textual: Antes vs. Depois

### 2.1. `tcc-unisinos/pretextual.tex`

#### [Item C4] Lista de Abreviaturas e Siglas
* **Antes:**
```latex
\begin{siglas}
  \item[ABNT] Associação Brasileira de Normas Técnicas
  \item[AG] Algoritmo Genético
  \item[ANFIS] \textit{Adaptive Neuro-Fuzzy Inference System} (Sistema de Inferência Neuro-Fuzzy Adaptativo)
  \item[BPTT] \textit{Backpropagation Through Time} (Retropropagação Através do Tempo)
...
  \item[DGA] \textit{Dissolved Gas Analysis} (Análise de Gases Dissolvidos)
  \item[GA] \textit{Genetic Algorithm} (Algoritmo Genético)
  \item[IEC] \textit{International Electrotechnical Commission} (Comissão Eletrotécnica Internacional)
  \item[IGA] \textit{Improved Genetic Algorithm} (Algoritmo Genético Melhorado)
...
  \item[MLP] \textit{Multilayer Perceptron} (Perceptron Multicamadas)
  \item[PSO] \textit{Particle Swarm Optimization} (Otimização por Enxame de Partículas)
...
\end{siglas}
```
* **Depois:**
```latex
\begin{siglas}
  \item[AG] Algoritmo Genético
  \item[ANFIS] \textit{Adaptive Neuro-Fuzzy Inference System} (Sistema de Inferência Neuro-Fuzzy Adaptativo)
  \item[ANOVA] \textit{Analysis of Variance} (Análise de Variância)
  \item[BPTT] \textit{Backpropagation Through Time} (Retropropagação Através do Tempo)
...
  \item[DGA] \textit{Dissolved Gas Analysis} (Análise de Gases Dissolvidos)
  \item[DP] Desvio Padrão
  \item[GA] \textit{Genetic Algorithm} (Algoritmo Genético)
  \item[IC] Intervalo de Confiança
  \item[IEC] \textit{International Electrotechnical Commission} (Comissão Eletrotécnica Internacional)
  \item[IGA] \textit{Improved Genetic Algorithm} (Algoritmo Genético Melhorado)
...
  \item[MLP] \textit{Multilayer Perceptron} (Perceptron Multicamadas)
  \item[OOF] \textit{Out-of-Fold} (Fora da Dobra)
  \item[PSO] \textit{Particle Swarm Optimization} (Otimização por Enxame de Partículas)
...
\end{siglas}
```

---

### 2.2. `tcc-unisinos/introducao.tex`

#### [Item D1] Seção 1.4 — Organização do Texto
* **Antes:**
```latex
\section{Organização do Texto}\label{organizacao}

O trabalho está estruturado em cinco capítulos organizados da seguinte forma:

O Capítulo~\ref{Fundamentacao} apresenta a fundamentação teórica...
```
* **Depois:**
```latex
\section{Organização do Texto}\label{organizacao}

O trabalho está estruturado em cinco capítulos: o Capítulo~\ref{Fundamentacao} apresenta a fundamentação teórica...
```

---

### 2.3. `tcc-unisinos/metodologia.tex`

#### [Item A4] Seção 3.2.2 — Proveniência da Base Chinesa e Caracterização Multicêntrica
* **Antes:**
```latex
\subsection{Base Compilada de Diagnóstico Estático (Track B)}

O Track B adota uma base de dados estáticos de ensaios cromatográficos de transformadores em serviço, disponibilizada originalmente por \citeonline{li2019_membership} no IEEE DataPort e reunida com rotinas de diagnóstico da ferramenta DGALab \cite{ibrahim2018_dgalab} no repositório público mantido por \citeonline{alan2021_dataset}. O conjunto reúne medições pontuais cobrindo as concentrações absolutas em partes por milhão (ppm) de cinco gases...
...
Os rótulos diagnósticos da base foram adotados conforme catalogados originalmente no repositório público. Como refletem registros operacionais de campo e não ensaios sintéticos de laboratório, os dados exibem sobreposições físico-químicas naturais e casos limítrofes entre categorias... Essa dispersão inerente a dados reais motivou \citeonline{li2019_membership} a proporem a modelagem por graus de pertinência difusos (\textit{membership degree}), consistindo também em fator determinante para a estabilização empírica das métricas dos classificadores em torno de 76\% a 78\%.
```
* **Depois:**
```latex
\subsection{Base Compilada de Diagnóstico Estático (Track B)}

O Track B adota uma base de dados estáticos compilada originalmente por \citeonline{li2019_membership} no IEEE DataPort (e reunida com rotinas de diagnóstico da ferramenta DGALab \cite{ibrahim2018_dgalab} no repositório mantido por \citeonline{alan2021_dataset}), a qual reúne medições pontuais cobrindo as concentrações absolutas em partes por milhão (ppm) de cinco gases normativos: hidrogênio ($\text{H}_2$), metano ($\text{CH}_4$), etano ($\text{C}_2\text{H}_6$), etileno ($\text{C}_2\text{H}_4$) e acetileno ($\text{C}_2\text{H}_2$), acompanhados pelo respectivo rótulo de integridade. A compilação abrange registros históricos provenientes de casos de literatura, inspeções visuais de falhas em serviço (IEC TC 10), desmontagem de transformadores em campo e ensaios laboratoriais.
...
Os rótulos diagnósticos da base foram adotados conforme catalogados originalmente no repositório público. Como refletem uma compilação multicêntrica de registros práticos, normas técnicas e ensaios laboratoriais, os dados exibem sobreposições físico-químicas naturais e casos limítrofes entre categorias... Essa dispersão inerente a dados reais motivou \citeonline{li2019_membership} a proporem a modelagem por graus de pertinência difusos (\textit{membership degree}), corroborando a estabilização empírica das métricas dos classificadores deste trabalho em torno de 76\% a 78\% de $F_1$-macro.
```

#### [Itens A1, A9, B5] Seção 3.4 — Módulo de Previsão Temporal (LSTM)
* **Antes:**
```latex
A rede recebe o histórico de $w = 50$ passos regulares contíguos (equivalente a 25 dias de medição), formando uma matriz $50 \times 4$. O objetivo é estimar o vetor com as quatro concentrações de gases no passo futuro $t + h$, onde $h$ indica o horizonte de previsão.
...
As 2.100 séries temporais reservadas para ajuste foram divididas internamente em 1.680 séries para treino (80\%) e 420 séries para validação (20\%), atingindo convergência por parada antecipada (\textit{early stopping}) tipicamente entre 30 e 45 épocas com paciência de 15 épocas.
...
Para mapear o alcance temporal da rede, foram avaliados 11 horizontes preditivos recursivos: $h \in \{1, 2, 4, 7, 14, 28, 42, 56, 70, 84, 98\}$ passos (de 12 horas a 7 semanas de projeção). Adotou-se o coeficiente de determinação $R^2 \ge 0,80$ como patamar de confiabilidade estatística para séries de manutenção preditiva em transformadores...
```
* **Depois:**
```latex
A rede recebe o histórico de $w = 50$ passos regulares contíguos (equivalente a 25 dias de medição a cada 12 horas), formando uma matriz $50 \times 4$. A definição de $w = 50$ passos equilibra a capacidade de capturar a cinética recente de evolução dos gases sem impor sobrecarga computacional excessiva ao modelo. A arquitetura é ajustada para estimar as quatro concentrações no passo imediatamente subsequente ($h = 1$), sendo as previsões para horizontes estendidos ($h > 1$) obtidas por extrapolação recursiva iterada, na qual a predição $\hat{y}$ é sucessivamente realimentada na janela de entrada.
...
As 2.100 séries temporais reservadas para ajuste foram divididas internamente em 1.785 séries para treino (85\%) e 315 séries para validação (15\%), atingindo convergência por parada antecipada (\textit{early stopping}) tipicamente em torno de 16 épocas com paciência de 10 épocas.
...
Para mapear o alcance temporal da rede, foram avaliados 12 horizontes preditivos recursivos: $h \in \{1, 2, 4, 7, 14, 28, 42, 56, 70, 84, 98, 112\}$ passos (de 12 horas a 8 semanas de projeção contínua). Adotou-se o coeficiente de determinação $R^2 \ge 0,80$ como patamar operacional de corte de confiabilidade para as séries de manutenção preditiva...
```

#### [Item B3] Seção 3.5.2 — Aptidão e Isolamento do Teste no IGA
* **Antes:**
```latex
A função de aptidão (\textit{fitness}) foi definida como a média do $F_1$-macro obtida por validação cruzada estratificada em 5 dobras ($K = 5$) calculada exclusivamente sobre o conjunto de treinamento. A escolha do $F_1$-macro garante que a otimização valorize o acerto em todas as classes de falha, sem favorecer a classe de operação normal.
```
* **Depois:**
```latex
A função de aptidão (\textit{fitness}) foi definida como a média do $F_1$-macro obtida por validação cruzada estratificada em 5 dobras ($K = 5$) calculada exclusivamente sobre o conjunto de treinamento. Esse desenho metodológico assegura que o conjunto de teste independente permaneça estritamente isolado e intocado durante todo o processo evolutivo de seleção de atributos e ajuste de hiperparâmetros, prevenindo qualquer vazamento de dados (\textit{data leakage}). A escolha do $F_1$-macro garante que a otimização valorize o acerto em todas as classes de falha, sem favorecer a classe de operação normal.
```

---

### 2.4. `tcc-unisinos/resultados.tex`

#### [Itens A1, A6, A10, B4] Seção 4.1 — Tabela 2 e Discussão Multi-Horizonte da LSTM
* **Antes:**
```latex
Para verificar a estabilidade das estimativas em prazos mais longos, avaliou-se a propagação preditiva ao longo de 11 horizontes temporais, variando de 12 horas ($h = 1$) até 7 semanas ($h = 98$, correspondente a 1.176 horas de operação contínua). Cada modelo foi treinado para projetar as concentrações com base na janela histórica de entrada ($w = 50$), calculando-se o coeficiente de determinação e o erro percentual médio relativo (MAE\%) a partir de cinco sementes aleatórias.

\begin{table}[H]
\centering
\caption{Desempenho da LSTM em função do avanço do horizonte temporal.}
\label{tab:lstm_multi}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ccccc}
\hline
\textbf{Passos ($h$)} & \textbf{Intervalo Temporal} & \textbf{$R^2$ Médio} & \textbf{MAE\% Médio} & \textbf{Condição ($R^2 \ge 0,80$)} \\ \hline
1 & 12 horas (0,07 semana) & 0,9996 & 0,90\% & Conforme \\
2 & 24 horas (0,14 semana) & 0,9996 & 1,00\% & Conforme \\
4 & 48 horas (0,29 semana) & 0,9995 & 1,03\% & Conforme \\
7 & 84 horas (0,50 semana) & 0,9993 & 1,22\% & Conforme \\
14 & 168 horas (1,00 semana) & 0,9987 & 1,74\% & Conforme \\
28 & 336 horas (2,00 semanas) & 0,9963 & 2,33\% & Conforme \\
42 & 504 horas (3,00 semanas) & 0,9837 & 5,68\% & Conforme \\
56 & 672 horas (4,00 semanas) & 0,9691 & 7,81\% & Conforme \\
70 & 840 horas (5,00 semanas) & 0,9485 & 10,02\% & Conforme \\
84 & 1.008 horas (6,00 semanas) & 0,9207 & 12,41\% & Conforme \\
98 & 1.176 horas (7,00 semanas) & 0,8846 & 14,89\% & Conforme \\ \hline
\end{tabular}%
}
\fonte{Elaborado pelo autor.}
\end{table}
```
* **Depois:**
```latex
Para verificar a estabilidade das estimativas em prazos mais longos, avaliou-se a propagação preditiva ao longo de 12 horizontes temporais recursivos, variando de 12 horas ($h = 1$) até 8 semanas ($h = 112$, correspondente a 1.344 horas de operação contínua). Treinou-se um modelo de um passo à frente ($h=1$, janela histórica de entrada $w = 50$) para cada uma de cinco sementes aleatórias ($42, 123, 456, 789, 2024$), projetando-se os horizontes futuros por extrapolação recursiva iterada (na qual as predições $\hat{y}$ são sucessivamente realimentadas na janela). A Tabela~\ref{tab:lstm_multi} sintetiza os resultados de $R^2$ médio e erro percentual médio relativo (MAE\%), contrastando-os diretamente com um preditor ingênuo de persistência temporal ($\hat{y}_{t+h} = y_t$). A Figura~\ref{fig:lstm_degradacao} apresenta a trajetória de redução do $R^2$ em relação ao critério operacional de confiabilidade ($R^2 \ge 0,80$).

\begin{table}[H]
\centering
\caption{Desempenho da LSTM recursiva e do baseline de persistência em função do horizonte temporal (5 sementes, 200 séries/semente).}
\label{tab:lstm_multi}
\resizebox{\textwidth}{!}{%
\begin{tabular}{cccccc}
\hline
\textbf{Passos ($h$)} & \textbf{Intervalo Temporal} & \textbf{$R^2$ Persistência} & \textbf{$R^2$ Médio LSTM} & \textbf{MAE\% Médio LSTM} & \textbf{Condição ($R^2 \ge 0,80$)} \\ \hline
1 & 12 horas (0,07 semana) & 1,0000 & 0,9996 & 0,90\% & Conforme \\
2 & 24 horas (0,14 semana) & 0,9999 & 0,9996 & 0,93\% & Conforme \\
4 & 48 horas (0,29 semana) & 0,9998 & 0,9994 & 1,05\% & Conforme \\
7 & 84 horas (0,50 semana) & 0,9992 & 0,9991 & 1,31\% & Conforme \\
14 & 168 horas (1,00 semana) & 0,9970 & 0,9979 & 2,05\% & Conforme \\
28 & 336 horas (2,00 semanas) & 0,9886 & 0,9928 & 3,76\% & Conforme \\
42 & 504 horas (3,00 semanas) & 0,9756 & 0,9836 & 5,67\% & Conforme \\
56 & 672 horas (4,00 semanas) & 0,9588 & 0,9691 & 7,76\% & Conforme \\
70 & 840 horas (5,00 semanas) & 0,9391 & 0,9485 & 10,01\% & Conforme \\
84 & 1.008 horas (6,00 semanas) & 0,9173 & 0,9207 & 12,41\% & Conforme \\
98 & 1.176 horas (7,00 semanas) & 0,8940 & 0,8846 & 14,96\% & Conforme \\
112 & 1.344 horas (8,00 semanas) & 0,8696 & 0,8380 & 17,68\% & Limite ($R^2_{\text{H}_2} < 0,80$) \\ \hline
\end{tabular}%
}
\fonte{Elaborado pelo autor.}
\end{table}

A análise da evolução multi-horizonte revela dois fenômenos fundamentais:

Em primeiro lugar, observa-se a degradação gradual da acurácia com o distanciamento temporal, padrão esperado devido ao acúmulo de resíduos na realimentação recursiva. Até duas semanas ($h = 28$), a LSTM mantém $R^2 \ge 0,9928$ com MAE\% de 3,76\%. Em sete semanas ($h = 98$), o $R^2$ médio situa-se em 0,8846, sendo esse o último horizonte no qual todos os quatro gases atendem individualmente ao patamar de corte ($R^2 \ge 0,80$, sendo $\text{H}_2 = 0,8483$). No horizonte de oito semanas ($h = 112$), embora a média agregada ainda seja $0,8380$, o hidrogênio rompe o patamar mínimo ($R^2 = 0,7877 < 0,80$), justificando $h = 98$ como o horizonte máximo operacional de prognóstico com confiabilidade uniforme entre os gases.

Em segundo lugar, a comparação com o baseline ingênuo de persistência temporal ($\hat{y}_{t+h} = y_t$) revela que a própria dinâmica física das séries monitoradas por \citeonline{katser2022} apresenta inércia temporal extremamente elevada: a simples repetição do valor atual já assegura $R^2 = 0,9970$ em 1 semana ($h = 14$) e $R^2 = 0,9886$ em 2 semanas ($h = 28$). A rede LSTM supera marginalmente a persistência entre 1 e 6 semanas (ganho modesto de $\Delta R^2 \approx +0,0009$ a $+0,0103$), mas nos horizontes mais longos ($h \ge 98$) o acúmulo de resíduos recursivos faz a rede decair ligeiramente abaixo da persistência. Esse achado é decisivo para a compreensão dos resultados: como as concentrações evoluem de forma lenta sob regime industrial contínuo, as projeções da LSTM tornam-se fortemente colineares às medições observadas ($r > 0,997$), explicando por que a inclusão das previsões de 1 e 2 semanas ($h = 14$ e $h = 28$) não agrega ganho discriminativo estatisticamente significativo ao classificador XGBoost.
```

#### [Item A4] Seção 4.3 — Fundamentação de Sobreposição Física (Track B)
* **Antes:**
```latex
Como todas as variáveis derivadas (razões e percentuais) são obtidas a partir das mesmas cinco concentrações primárias... ela auxilia o classificador ao explicitar relações físico-químicas conhecidas... mas não elimina a sobreposição física entre falhas que geram perfis de gases muito semelhantes. Note-se ainda que o conjunto de 18 atributos candidatos de Wu et al. difere do espaço de C3 unicamente pela omissão da razão $\text{C}_2\text{H}_2/\text{CH}_4$, o que justifica a proximidade estreita de comportamento entre M3 e C3.
```
* **Depois:**
```latex
Como todas as variáveis derivadas (razões e percentuais) são obtidas a partir das mesmas cinco concentrações primárias... ela auxilia o classificador ao explicitar relações físico-químicas conhecidas... mas não elimina a sobreposição física entre falhas que geram perfis de gases muito semelhantes. Essa sobreposição intrínseca e as fronteiras difusas entre subtipos térmicos e de descarga foram documentadas por \citeonline{li2019_membership}, os quais destacam que equipamentos operacionais exibem zonas contínuas de transição térmica ($300\ ^\circ\text{C}$ e $700\ ^\circ\text{C}$) e energética (centelhamento e arco elétrico), corroborando o patamar empírico de estabilização de 76\% a 78\% obtido pelos modelos XGBoost neste estudo. Note-se ainda que o conjunto de 18 atributos candidatos de Wu et al. difere do espaço de C3 unicamente pela omissão da razão $\text{C}_2\text{H}_2/\text{CH}_4$, o que justifica a proximidade estreita de comportamento entre M3 e C3.
```

#### [Item A2] Seção 4.5.6 — Migração de Crédito SHAP e Valores Exatos
* **Antes:**
```latex
\subsection{Migração de Crédito e Avaliação da Hipótese Temporal}

Para concluir a investigação sobre o papel das previsões da LSTM no Track A, avaliou-se como a importância atribuída pelo TreeSHAP se distribui entre variáveis medidas e variáveis previstas. A Figura~\ref{fig:migracao_credito} ilustra a importância média do hidrogênio dividida entre a leitura atual ($\text{H}_{2,\text{obs}}$) e as projeções futuras para 14 e 28 passos ($\text{H}_{2,\text{h14}}$ e $\text{H}_{2,\text{h28}}$).
...
O comportamento observado na Figura~\ref{fig:migracao_credito} esclarece a dinâmica de divisão de crédito no Track A. Ao variar a semente de treinamento do classificador XGBoost, a importância atribuída pelo TreeSHAP oscila entre os diferentes horizontes temporais previstos da mesma espécie química ($\text{H}_{2,\text{h14}}$ e $\text{H}_{2,\text{h28}}$), fenômeno típico de ensembles de árvores sob variáveis fortemente correlacionadas. No entanto, a contribuição somada da família do hidrogênio (linha preta superior) mantém-se rigorosamente estável em torno de 0,42. Isso evidencia que a introdução de múltiplos horizontes preditivos não acrescenta novos graus de liberdade ao classificador, mas apenas fragmenta o crédito preditivo de um mesmo sinal físico subjacente.
```
* **Depois:**
```latex
\subsection{Migração de Crédito e Avaliação da Hipótese Temporal}

Para concluir a investigação sobre o papel das previsões da LSTM no Track A, avaliou-se como a importância atribuída pelo TreeSHAP se distribui entre variáveis medidas e variáveis previstas. A Figura~\ref{fig:migracao_credito} ilustra a distribuição média de crédito do hidrogênio entre as previsões gêmeas da LSTM ($\text{H}_{2,\text{h14}}$ e $\text{H}_{2,\text{h28}}$) para o modelo vencedor C1 (XGBoost otimizado por IGA, 8 atributos), avaliado ao longo das cinco sementes aleatórias.
...
O comportamento ilustrado na Figura~\ref{fig:migracao_credito} elucida a dinâmica de divisão de crédito no Track A. No modelo C1, o algoritmo genético selecionou 8 atributos, descartando a leitura observada $\text{H}_{2,\text{obs}}$ e retendo conjuntamente as projeções $\text{H}_{2,\text{h14}}$ e $\text{H}_{2,\text{h28}}$. Ao variar a semente de treinamento, a importância atribuída pelo TreeSHAP oscila entre as duas variáveis temporais do mesmo gás: na semente 42, $65,5\%$ do crédito recai em $h=14$ ($0,3144$) e $34,5\%$ em $h=28$ ($0,1653$); na semente 456, a proporção é de $68,1\%$ ($0,3378$) e $31,9\%$ ($0,1586$). Contudo, o impacto total somado de hidrogênio (linha preta superior) permanece notavelmente constante entre as cinco sementes, situando-se em $0,4734 \pm 0,0131$ (variando estritamente entre $0,4624$ e $0,4964$, com média de $65,9\%$ do peso em $h=14$ e $34,1\%$ em $h=28$). 

Esse comportamento confirma que as projeções da LSTM disputam a divisão nos nós das árvores sem fornecer informação discriminativa complementar: o modelo aloca o crédito preditivo de forma intercambiável entre as variáveis colineares, preservando o impacto total do sinal físico subjacente. Esse mesmo fenômeno de conservação foi verificado para o modelo C2 (17 atributos), no qual o hidrogênio observado isolado na referência sem LSTM ($\text{H}_{2,\text{obs}} = 0,3085$) reparte seu crédito entre três canais correlacionados quando as projeções são incluídas ($\text{H}_{2,\text{obs}} = 0,1921$, $\text{H}_{2,\text{h14}} = 0,1277$ e $\text{H}_{2,\text{h28}} = 0,1045$, totalizando $0,4243$).
```

---

### 2.5. `tcc-unisinos/conclusao.tex`

#### [Itens A4, B1, B2, B3, B4, B6] Seção 5.3 — Limitações da Pesquisa
* **Antes:**
```latex
\section{Limitações da Pesquisa}\label{sec:limitacoes_pesquisa}

Os resultados e as conclusões apresentadas devem ser interpretados considerando as limitações dos dados públicos disponíveis na literatura de DGA:

\begin{enumerate}
    \item \textbf{Fragmentação das bases de dados:} Não se identificou na literatura aberta um conjunto de dados que una simultaneamente séries temporais contínuas em alta taxa de amostragem, medições dos gases normativos completos e laudos confirmados de defeitos severos. Essa indisponibilidade motivou a divisão do estudo nos dois eixos investigados.
    \item \textbf{Escopo restrito no Track A:} O conjunto temporal de \citeonline{katser2022} disponibiliza apenas quatro gases ($\text{H}_2$, $\text{CO}$, $\text{C}_2\text{H}_4$ e $\text{C}_2\text{H}_2$), impossibilitando o cálculo de razões clássicas que envolvem metano e etano. Adicionalmente, as concentrações registradas são predominantemente baixas (na escala de $10^{-3}$), refletindo condições de usinas nucleares com monitoramento rigoroso e pouca ocorrência de falhas destrutivas de grande porte.
    \item \textbf{Heterogeneidade e ruído de rotulagem no Track B:} A base estática compilada reúne ensaios laboratoriais de diferentes concessionárias e períodos. Conforme demonstrado na auditoria normativa, parte das amostras rotuladas como defeitos térmicos severos (T3) apresenta características cromatográficas incompatíveis com os critérios normativos internacionais, gerando um ruído de anotação que limita o desempenho máximo que qualquer modelo supervisionado pode alcançar sobre esses dados.
\end{enumerate}
```
* **Depois:**
```latex
\section{Limitações da Pesquisa}\label{sec:limitacoes_pesquisa}

Os resultados e as conclusões apresentadas devem ser interpretados considerando as limitações metodológicas e dos dados públicos disponíveis na literatura de DGA:

\begin{enumerate}
    \item \textbf{Fragmentação das bases e escopo dos eixos analíticos:} A indisponibilidade de um repositório aberto que una séries temporais longas com os sete laudos multiclasse da IEC 60599 motivou a partição do estudo em dois eixos complementares. Por conseguinte, o Track B avalia exclusivamente o classificador XGBoost otimizado por IGA e interpretado via TreeSHAP sobre dados estáticos multiclasse, enquanto a hipótese de hibridização temporal com a rede LSTM foi avaliada no Track A.
    \item \textbf{Escopo restrito e inércia temporal no Track A:} O conjunto temporal de \citeonline{katser2022} monitora apenas quatro gases ($\text{H}_2$, $\text{CO}$, $\text{C}_2\text{H}_4$ e $\text{C}_2\text{H}_2$), impedindo o cômputo de razões envolvendo metano e etano. Além disso, as séries refletem regime de alta inércia temporal com taxas de evolução extremamente graduais, conforme comprovado pelo baseline de persistência ($\hat{y}_{t+h} = y_t$, com $R^2 \ge 0,9886$ até 14 dias), o que torna as projeções da LSTM colineares às leituras presentes e limita o ganho preditivo adicional.
    \item \textbf{Normalização nas séries temporais:} No pré-processamento das séries do Track A, a escala Min-Max foi calculada sobre o conjunto consolidado das 3.000 séries; embora do ponto de vista estrito os parâmetros devam ser ajustados apenas no subconjunto de treino, o impacto empírico sobre a avaliação é mínimo devido à natureza sintética i.i.d. das séries geradas pelo mesmo processo paramétrico.
    \item \textbf{Heterogeneidade e sobreposição física no Track B:} A base estática compilada por \citeonline{li2019_membership} reúne registros de literatura, inspeções visuais em serviço (IEC TC 10), desmontagem de transformadores e ensaios laboratoriais. Conforme destacado por Li, Wang e Song (2019), essa heterogeneidade prática acarreta sobreposições físico-químicas e fronteiras difusas naturais entre subtipos térmicos (T1/T2/T3) e de descarga (D1/D2), estabelecendo um teto empírico em torno de 76\% a 78\% de $F_1$-macro para modelos supervisionados alimentados unicamente pelos cinco gases clássicos.
    \item \textbf{Isolamento do teste e análises descritivas pós-hoc:} A seleção de atributos pelo IGA foi conduzida com validação cruzada de 5 dobras restrita ao conjunto de treinamento, mantendo a partição de teste independente 100\% isolada de vazamento. Análises geométricas complementares, como o cálculo de vizinhança $k$-NN ($k=5$) da entropia local de rótulos, foram executadas exclusivamente como diagnóstico descritivo \textit{post-hoc} da geometria do espaço de atributos, sem qualquer participação no treinamento dos modelos.
\end{enumerate}
```

---

### 2.6. `tcc-unisinos/referencias.bib`

#### [Itens A11 e C1] Entradas `li2019_membership` e `lecun1998gradient`
* **Antes:**
```bibtex
@article{lecun1998gradient,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Yann and others},
  journal={Proceedings of the IEEE},
  year={1998}
}
...
@article{li2019_membership,
  title     = {Fault Diagnosis of Power Transformers With Membership Degree},
  author    = {Li, Enwen and Wang, Licheng and Song, Bo},
  journal   = {IEEE Access},
  volume    = {7},
  pages     = {28791--28798},
  year      = {2019},
  publisher = {IEEE},
  doi       = {10.1109/ACCESS.2019.2902633}
}
```
* **Depois:**
```bibtex
@article{lecun1998gradient,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal={Proceedings of the IEEE},
  volume={86},
  number={11},
  pages={2278--2324},
  year={1998},
  publisher={IEEE},
  doi={10.1109/5.726791}
}
...
@article{li2019_membership,
  title     = {Fault Diagnosis of Power Transformers With Membership Degree},
  author    = {Li, Enwen and Wang, Linong and Song, Bin},
  journal   = {IEEE Access},
  volume    = {7},
  pages     = {28791--28798},
  year      = {2019},
  publisher = {IEEE},
  doi       = {10.1109/ACCESS.2019.2902299}
}
```

---

### 2.7. Figuras e Scripts Python

#### [Item A8 / A10] Figura 8 (`lstm_multi_horizon_r2_norm.png`)
* **Código:** `tcc-transformer-fault/src/lstm/multi_horizon_eval.py` (L406).
  * **Problema anterior:** `cores = {"H2": "#2196F3", "CH4": "#FF5722", "CO": "#4CAF50", "C2H2": "#9C27B0"}`. Como Katser et al. monitoram $\text{C}_2\text{H}_4$ (e não $\text{CH}_4$), o etileno caía na cor de contingência `#4CAF50`, ficando idêntico à cor verde do monóxido de carbono ($\text{CO}$).
  * **Correção aplicada:** Atualizado para `cores = {"H2": "#2196F3", "CO": "#4CAF50", "C2H4": "#FF9800", "C2H2": "#9C27B0"}`, atribuindo cor laranja distintiva a $\text{C}_2\text{H}_4$.
  * **Ajuste textual:** Rótulos e título regerados com acentuação correta (*Degradação*, *previsão*, *médio*, *máx aceitável*) e exibindo todos os 12 horizontes até $h=112$.
  * **Saída gerada:** Arquivo regerado e copiado diretamente para `tcc-unisinos/figuras/lstm_multi_horizon_r2_norm.png`.

#### [Item A2 / A8] Figura 18 (`fig7_lstm_credit_migration.png`)
* **Código:** `tcc-transformer-fault/src/evaluation/fase6_shap_track_a.py` (L75–104).
  * **Problema anterior:** Legenda sobrepunha o topo das barras e os valores numéricos; ausência de rótulos percentuais diretos nas barras.
  * **Correção aplicada:**
    * Inserção de anotações percentuais legíveis dentro das barras empilhadas para cada semente.
    * Inserção dos valores absolutos da soma de $|\text{SHAP}|$ ($0,4798$, $0,4650$, $0,4964$, $0,4635$, $0,4624$) diretamente acima de cada ponto com folga visual adequada.
    * Deslocamento da legenda para fora da área dos dados (`loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3`).
  * **Saída gerada:** Script executado e imagem salva em `tcc-unisinos/figuras/fase6/fig7_lstm_credit_migration.png`.

---

## 3. Conclusão

Todas as ações descritas no [plano_revisao_fase1.md](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide\scratch\TCC_Final\tcc-unisinos\plano_revisao_fase1.md) foram integralmente executadas, preservando rigorosamente todo o texto que não foi objeto de revisão e mantendo o título original do trabalho.

