# Plano de Revisão — Fase 1 (Consolidado e Pronto para Execução)

> **Status do Plano:** Todas as informações técnicas foram extraídas/verificadas diretamente nos dados e códigos de `tcc-transformer-fault/` e `tcc-unisinos/artigos/`, e todas as decisões editoriais foram definidas pelo autor. **Nenhuma pendência aberta resta.** Aguardando apenas o comando para iniciar a execução.

---

## Decisões Editoriais Consolidadas pelo Autor

1. **Horizonte $h=112$ (8 semanas / 560 h) na Tabela 2 e Seção 4.1:** **APROVADO (Opção 1A).** A Tabela 2 (`tab:lstm_multi`) e o texto da Seção 4.1 serão atualizados com os 12 horizontes oficiais de `reports/metrics/lstm_multi_horizon_results_norm.json` (incluindo $h=112$, onde o $R^2$ médio é $0,8380$, mas o gás individual $\text{H}_2$ cai para $0,7877 < 0,80$, justificando o limite operacional em $h=98$ / 7 semanas para todos os gases individualmente).
2. **Baseline de Persistência ($\hat{y}_{t+h} = y_t$):** **APROVADO (Opção 2A).** Os valores do baseline de persistência (calculados sobre `katser_test.npz` com o mesmo protocolo de 5 sementes $\times$ 200 séries) serão incluídos na discussão de Resultados (Seções 4.1/4.2) e mencionados na Seção 5.3, demonstrando quantitativamente a alta inércia temporal das séries ($R^2_{\text{persist}} = 0,9970$ em $h=14$ e $0,9886$ em $h=28$) e explicando por que as projeções da LSTM são colineares ao estado observado $y_t$.
3. **Regeração das Figuras 7, 8 e 18:** **APROVADO.** As três figuras serão corrigidas diretamente a partir dos seus arquivos-fonte (`Arquitetura.svg`, `multi_horizon_eval.py` e `fase6_shap_track_a.py`) e reexportadas em PDF.
4. **Título do TCC (`pretextual.tex`):** **MANTIDO O TÍTULO ORIGINAL.** Nenhuma alteração será feita no título do trabalho.

---

## Bloco A — Consistência Interna, Dados Oficiais e Correções Factuais

### A1. Previsão Recursiva da LSTM (`metodologia.tex` vs. `resultados.tex`)
- **Onde:** `metodologia.tex` (L88) e `resultados.tex` (L28–31).
- **Base no código:** Em `src/lstm/multi_horizon_eval.py` (`previsao_recursiva`, L82–107) e `src/feature_engineering/build_katser_features.py` (`prever_gases_futuros`, L142–175), a arquitetura treina um único modelo de 1 passo à frente (`LSTM_HORIZONTE = 1`, `Dense(4)`) para cada uma das 5 sementes (`[42, 123, 456, 789, 2024]`) e extrapola para $h > 1$ de forma **100% recursiva (iterada)**, realimentando a janela deslizante de 50 passos com as próprias predições $\hat{y}$.
- **Ação:** Reescrever `resultados.tex` (L28–31) deixando explícito que os 5 modelos treinados (um por semente para $h=1$) foram avaliados por extrapolação recursiva nos 12 horizontes futuros (de $h=1$ a $h=112$).

### A2. Alinhamento da Figura 18, Seção 4.5.2 e Seção 4.5.6 (Migração de Crédito SHAP)
- **Onde:** `resultados.tex` (L259, L273–275 e L320–329).
- **Base no código (`src/evaluation/fase6_shap_track_a.py`):**
  - Na Figura 18, o gráfico exibe o modelo vencedor **C1 (XGBoost + IGA, 8 atributos)**, no qual o IGA descartou `obs_H2` e selecionou apenas `lstm_h14_H2` e `lstm_h28_H2`. Já a comparação com `obs_H2` na Seção 4.5.2 refere-se aos modelos completos sem poda (`B1_default` vs. `C2_default`).
  - Valores exatos de $|\text{SHAP}|$ médio em C1 nas 5 sementes:
    - **Seed 42:** `lstm_h14_H2` $= 0,3144$ ($65,5\%$) | `lstm_h28_H2` $= 0,1653$ ($34,5\%$) | **Soma $= 0,4798$**
    - **Seed 123:** `lstm_h14_H2` $= 0,3150$ ($67,7\%$) | `lstm_h28_H2` $= 0,1500$ ($32,3\%$) | **Soma $= 0,4650$**
    - **Seed 456:** `lstm_h14_H2` $= 0,3378$ ($68,1\%$) | `lstm_h28_H2` $= 0,1586$ ($31,9\%$) | **Soma $= 0,4964$**
    - **Seed 789:** `lstm_h14_H2` $= 0,2970$ ($64,1\%$) | `lstm_h28_H2` $= 0,1665$ ($35,9\%$) | **Soma $= 0,4635$**
    - **Seed 2024:** `lstm_h14_H2` $= 0,2953$ ($63,9\%$) | `lstm_h28_H2` $= 0,1671$ ($36,1\%$) | **Soma $= 0,4624$**
    - **Média $\pm$ DP (5 seeds):** `lstm_h14_H2` $= 0,3119 \pm 0,0154$ ($65,9\%$), `lstm_h28_H2` $= 0,1615 \pm 0,0066$ ($34,1\%$), **Soma $= 0,4734 \pm 0,0131$** (faixa de $0,4624$ a $0,4964$).
- **Ação:** Atualizar a Seção 4.5.6 (`resultados.tex` L320–329) com os valores exatos de C1 ($0,4624$ a $0,4964$, média de $0,4734 \pm 0,0131$) e explicar de forma transparente a diferença entre o subconjunto selecionado pelo IGA em C1 (Figura 18, sem $\text{H}_{2,\text{obs}}$) e o conjunto completo de 17 atributos em C2 (Seção 4.5.2, onde a soma das três versões de $\text{H}_2$ é $0,1921 + 0,1277 + 0,1045 = 0,4243$).

### A3. Citação de Wu, Shu e Li (2021) em `resultados.tex` (L163)
- **Status:** Já corrigido em commit anterior (texto atual já cita *125 amostras de treino e 25 de teste* e formula a questão de duplicatas como hipótese). Nenhuma alteração necessária.

### A4. Proveniência da Base Chinesa e Citação de Li, Wang e Song (2019)
- **Onde:** `metodologia.tex` (L33, L42), `resultados.tex` (L169) e `conclusao.tex` (L35).
- **Base no PDF (`Fault_Diagnosis_of_Power_Transformers_With_Membership_Degree.pdf`, p. 2, 4 e 5):**
  1. Os dados compilados por Li, Wang e Song (2019) reúnem registros históricos da literatura, inspeções visuais de falhas em serviço (IEC TC 10), desmontagens de transformadores em campo (*State Grid Corporation of China*), casos da norma DL/T722-2014 e ensaios simulados em laboratório (Duval, 2002).
  2. Os autores motivam o uso de graus de pertinência (*membership degree*) explicitamente pela sobreposição física e fronteira difusa (*fuzzy transitional area*) entre os subtipos térmicos T1/T2/T3 ($300\ ^\circ\text{C}$ e $700\ ^\circ\text{C}$) e de descarga D1/D2.
- **Ação:**
  - Em `metodologia.tex` (L33 e L42): descrever a base como compilação multicêntrica (literatura, registros de inspeção em campo e ensaios laboratoriais) reportada por Li, Wang e Song (2019).
  - Em `resultados.tex` (L169) e `conclusao.tex` (L35): manter a citação a Li, Wang e Song (2019) para fundamentar a sobreposição física e fronteira difusa entre T1/T2/T3 e D1/D2, apresentando o teto empírico de $76\text{--}78\%$ de F1 Macro no XGBoost como constatação experimental deste TCC (corroborada pela sobreposição descrita pelos autores).

### A5. Norma de 6 Classes na Trilha A (IEC 60599)
- **Onde:** `metodologia.tex` (L38).
- **Ação:** Substituir `IEEE C57.104` por `IEC 60599` para manter consistência com a Tabela 1 (`metodologia.tex` L49) e com Katser et al. (2021).

### A6. Critério Operacional $R^2 \ge 0,80$ para Horizonte Máximo da LSTM
- **Onde:** `resultados.tex` (L34).
- **Ação:** Substituir *"critério de qualidade preditiva aceitável em prognóstico DGA ($R^2 \geq 0,80$)"* por *"critério operacional adotado neste trabalho ($R^2 \geq 0,80$)"*.

### A7. Métrica na Conclusão (F1 Macro vs. Acurácia Global)
- **Onde:** `conclusao.tex` (L23).
- **Ação:** Substituir *"acurácia global"* por *"F1 Macro"* ao reportar os valores de $69,2\%$ (Trilha A) e $76,4\%$ (Trilha B).

### A8. Regeração das Figuras 7, 8 e 18
- **Figura 7 (`tcc-unisinos/figuras/Arquitetura.svg` $\rightarrow$ `Arquitetura.pdf`):**
  - Corrigir `"84,3% (Remoção)"` para `"84,7% (Remoção)"` (referente às 500 de 590 duplicatas exatas removidas) e `"Split Estratificado (64/16/20)"` para `"Split Estratificado (68/12/20)"` (1.578 / 278 / 465 amostras).
  - Harmonizar o rótulo `"5 Gases Comuns"` com o texto da Seção 3.1.
- **Figura 8 (`src/lstm/multi_horizon_eval.py` $\rightarrow$ `fig_lstm_r2_degradacao.pdf`):**
  - Corrigir na linha 406 de `multi_horizon_eval.py` a chave do dicionário de cores (`"C2H4"` em vez de `"CH4"`), eliminando a duplicação da cor verde entre $\text{C}_2\text{H}_4$ e $\text{CO}$.
  - Restaurar a acentuação em português no título e na legenda (`"Degradação"`, `"Média"`) e regerar o PDF com todos os 12 horizontes (até $h=112$).
- **Figura 18 (`src/evaluation/fase6_shap_track_a.py` $\rightarrow$ `fig_shap_lstm_credit_migration.pdf`):**
  - Ajustar o posicionamento da legenda e os limites do eixo vertical em `plot_lstm_credit_migration` (L60–102) para que a legenda não sobreponha as barras nem os valores de soma no topo, regerando o PDF.

### A9. Proporção de Treino/Validação e Épocas da LSTM em `metodologia.tex`
- **Onde:** `metodologia.tex` (L99).
- **Base no código:** `preprocess_katser.py` (`SPLIT_VAL_KATSER = 0.15`), `analise_resultados_lstm_fase3.md` (L26–27) e `iga_track_a_results.json` (L67).
- **Ação:** Corrigir de *"1.680 séries para treino (80%) e 420 séries para validação (20%), atingindo convergência tipicamente entre 30 e 45 épocas"* para **1.785 séries para treino (85%) e 315 séries para validação (15%), atingindo convergência tipicamente em torno de 16 épocas**.

### A10. Atualização da Tabela 2 (`tab:lstm_multi`) e Seção 4.1 com `lstm_multi_horizon_results_norm.json` (Incluindo $h=112$)
- **Onde:** `resultados.tex` (L29–58, Tabela 2 e texto da Seção 4.1).
- **Ação:** Atualizar a Tabela 2 e o texto para apresentar os 12 horizontes oficiais de `reports/metrics/lstm_multi_horizon_results_norm.json`, incluindo a linha $h=112$ (8 semanas / 560 h) onde $\text{H}_2$ cruza abaixo do limiar de $0,80$:
  - $h=1$ (5 h): $R^2 = 0,9996$, MAE% $= 0,90\%$
  - $h=2$ (10 h): $R^2 = 0,9996$, MAE% $= 0,93\%$
  - $h=4$ (20 h): $R^2 = 0,9994$, MAE% $= 1,05\%$
  - $h=7$ (35 h / 0,5 sem.): $R^2 = 0,9991$, MAE% $= 1,31\%$
  - $h=14$ (70 h / 1 sem.): $R^2 = 0,9979$, MAE% $= 2,05\%$
  - $h=28$ (140 h / 2 sem.): $R^2 = 0,9928$, MAE% $= 3,76\%$
  - $h=42$ (210 h / 3 sem.): $R^2 = 0,9836$, MAE% $= 5,67\%$
  - $h=56$ (280 h / 4 sem.): $R^2 = 0,9691$, MAE% $= 7,76\%$
  - $h=70$ (350 h / 5 sem.): $R^2 = 0,9485$, MAE% $= 10,01\%$
  - $h=84$ (420 h / 6 sem.): $R^2 = 0,9207$, MAE% $= 12,41\%$
  - $h=98$ (490 h / 7 sem.): $R^2 = 0,8846$, MAE% $= 14,96\%$ (último horizonte com todos os 4 gases $\ge 0,80$, sendo $\text{H}_2 = 0,8483$)
  - $h=112$ (560 h / 8 sem.): $R^2 = 0,8380$, MAE% $= 17,68\%$ ($\text{H}_2$ cai para $0,7877 < 0,80$).

### A11. Correção Bibliográfica de Li, Wang e Song (2019) em `referencias.bib`
- **Onde:** `referencias.bib` (L311–320, chave `li2019`).
- **Ação:** Corrigir `author` para `{Li, Enwen and Wang, Linong and Song, Bin}` e `doi` para `{10.1109/ACCESS.2019.2902299}` conforme o PDF original.

---

## Bloco B — Rigor Metodológico (Seções 3.5, 4.1, 4.2 e 5.3)

| # | Ponto | Onde | Ação Consolidada |
|---|---|---|---|
| **B1** | Normalização global na Trilha A (`preprocess_katser.py`) | `conclusao.tex` (Seção 5.3) | Documentar que o `StandardScaler` das 3.000 séries sintéticas de Katser foi ajustado no tensor completo antes da partição treino/teste, observando que o impacto prático é desprezível por se tratar de séries sintéticas i.i.d. oriundas do mesmo gerador paramétrico. |
| **B2** | $k$-NN ($k=5$) da entropia de rótulo local sobre as 2.321 amostras (`fase6_shap_extras.py`) | `conclusao.tex` (Seção 5.3) | Explicitar que o cálculo de vizinhança da Figura 16 foi executado sobre a base consolidada de 2.321 amostras exclusivamente como diagnóstico descritivo *post-hoc* da geometria do espaço de atributos (sem qualquer participação no treinamento ou seleção de modelos). |
| **B3** | Seleção de atributos pelo IGA via CV 5-fold no conjunto de treino | `metodologia.tex` / `conclusao.tex` | Destacar que a busca do IGA ocorreu com validação cruzada estratificada de 5 *folds* restrita ao conjunto de treino (mantendo o conjunto de teste 100% isolado de vazamento), e registrar na Seção 5.3 a natureza *wrapper* dessa etapa interna. |
| **B4** | **Baseline de Persistência ($\hat{y}_{t+h} = y_t$)** | `resultados.tex` (Seções 4.1 e 4.2) e `conclusao.tex` (Seção 5.3) | Inserir na Tabela 2 (ou no texto da Seção 4.1/4.2) a comparação direta entre a LSTM e o preditor ingênuo de persistência ($\hat{y}_{t+h} = y_t$) sobre `katser_test.npz`:<br>• $h=1$: $R^2_{\text{persist}} = 1,0000$ vs. $R^2_{\text{LSTM}} = 0,9996$<br>• $h=7$: $R^2_{\text{persist}} = 0,9992$ vs. $R^2_{\text{LSTM}} = 0,9991$<br>• $h=14$: $R^2_{\text{persist}} = 0,9970$ vs. $R^2_{\text{LSTM}} = 0,9979$<br>• $h=28$: $R^2_{\text{persist}} = 0,9886$ vs. $R^2_{\text{LSTM}} = 0,9928$<br>• $h=56$: $R^2_{\text{persist}} = 0,9588$ vs. $R^2_{\text{LSTM}} = 0,9691$<br>• $h=98$: $R^2_{\text{persist}} = 0,8940$ vs. $R^2_{\text{LSTM}} = 0,8846$<br>• $h=112$: $R^2_{\text{persist}} = 0,8696$ vs. $R^2_{\text{LSTM}} = 0,8380$<br>Discutir como essa altíssima inércia temporal ($R^2_{\text{persist}} \ge 0,9886$ até 2 semanas) comprova por que as projeções da LSTM são quase colineares a $y_t$ e não agregam ganho discriminativo ao XGBoost. |
| **B5** | Justificativa do corte $t=100$ e janela $w=50$ na Trilha A | `metodologia.tex` (Seção 3.5) | Explicitar que $w=50$ (250 horas) captura a dinâmica recente da série sem custo quadrático excessivo e que o corte fixo em $t=100$ (metade dos 200 passos da série) garante histórico suficiente ($t \ge 50$) e folga futura de 100+ passos para avaliar a extrapolação multi-horizonte. |
| **B6** | Escopo estático da Trilha B | `metodologia.tex` e `conclusao.tex` | Reforçar que a Trilha B (base real estática, sem séries temporais) valida o subsistema de classificação, seleção evolutiva (IGA) e explicabilidade (SHAP), enquanto a hipótese de hibridização temporal com a LSTM é avaliada na Trilha A. |

---

## Bloco C — Higiene ABNT / LaTeX

| # | Onde | Problema | Correção |
|---|---|---|---|
| **C1** | `referencias.bib` (L275, L283, L294, L302) | `and others` literal em `katser2021`, `castro2023`, `wu2021`, `paterakis2017` | Substituir pelos nomes completos dos coautores (conforme levantado em `auditoria_referencias.md`). |
| **C2** | `referencias.bib` (L261) | `author = {IEEE}` em `ieec57104` gerando `(IEEE, 2019)` | Ajustar entrada para o formato ABNT de normas técnicas. |
| **C3** | `metodologia.tex` (L190) | `\cite{lundberg2017}` dentro de `\caption{}` da Figura 7 | Substituir por texto plano `Lundberg e Lee (2017)` dentro do `\caption` (ou usar argumento opcional `\caption[...]`). |
| **C4** | `pretextual.tex` (L128, 134, 146) | Siglas `ABNT`, `F1`, `GDA` sem correspondência exata ou órfãs | Remover `ABNT`, `F1`, `GDA` e incluir `ANOVA`, `DP`, `IC`, `OOF`. |
| **C5** | `resultados.tex` (L214) | Acento agudo solto (`´`) em vez de sinal de menos (`-0,0067`) | Substituir `´0,0067` por `-0,0067`. |

---

## Bloco D — Polimento de Texto

| # | Onde | Problema | Correção |
|---|---|---|---|
| **D1** | `introducao.tex` (L49) | Parágrafo de uma frase só | Unir ao parágrafo anterior. |
| **D2** | `introducao.tex` (L59–61) | Repetição de "lacuna" | Variar vocabulário na segunda ocorrência. |
| **D3** | `resultados.tex` (L314, L347) | Anglicismos sem itálico ("trade-off", "deployment") | Colocar `\textit{trade-off}` e substituir "deployment" por "implantação em campo". |

---

## Ordem de Execução (Quando Autorizado)

1. **Etapa 1 — Regeração das Figuras 7, 8 e 18 (A8):**
   - Atualizar `tcc-unisinos/figuras/Arquitetura.svg` e exportar `Arquitetura.pdf`.
   - Corrigir cores e acentuação em `src/lstm/multi_horizon_eval.py` e regerar `tcc-unisinos/figuras/fig_lstm_r2_degradacao.pdf`.
   - Ajustar legenda e limites em `src/evaluation/fase6_shap_track_a.py` e regerar `tcc-unisinos/figuras/fig_shap_lstm_credit_migration.pdf`.
2. **Etapa 2 — Atualização Bibliográfica (`referencias.bib` — A11, C1, C2):**
   - Corrigir prenomes e DOI de `li2019`, expandir `and others` em `katser2021`, `castro2023`, `wu2021`, `paterakis2017`, e padronizar `ieec57104`.
3. **Etapa 3 — Revisão dos Capítulos LaTeX (`pretextual.tex`, `introducao.tex`, `metodologia.tex`, `resultados.tex`, `conclusao.tex`):**
   - Aplicar os itens A1, A2, A4–A7, A9, A10, B1–B6, C3–C5 e D1–D3 (mantendo o título original em `pretextual.tex`).
4. **Etapa 4 — Compilação e Verificação do PDF (`artigo.pdf`):**
   - Compilar com `latexmk` / `pdflatex` + `bibtex` e verificar que não há erros de LaTeX, referências indefinidas ou sobreposições visuais.
