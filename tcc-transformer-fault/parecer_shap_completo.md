# Parecer Técnico Completo: Avaliação SHAP (Fase 6)

**Avaliador:** Especialista Sênior em XAI / Interpretabilidade de Modelos  
**Data:** 07/10/2026  
**Escopo:** Avaliação crítica do documento `interpretacao_shap_fase6.md`, dos 14 gráficos, das 2 métricas, e dos scripts geradores. Análise cruzada com `parecer_banca_fase6.md` e `relatorio_decisao_fase6.md`.

---

## 1. Avaliação Crítica do Documento

### 1.1 Veredito Geral

O documento `interpretacao_shap_fase6.md` é **bom como rascunho de escrita**, mas tem **quatro problemas técnicos que a banca pode explorar**. A linguagem acadêmica está, no geral, adequada para um TCC de engenharia — porém há trechos que exageram a conclusão além do que os dados suportam.

### 1.2 O que está CORRETO e SÓLIDO

| Ponto | Avaliação |
|---|---|
| **Teste de estabilidade (ρ = 0.963)** | ✅ Metodologicamente correto. Spearman do ranking de \|φ\| médio entre 5 seeds é uma boa prática. ρ > 0.8 é um limiar razoável para interpretar a ordem fina. |
| **Agrupamento por famílias (Fig. 2)** | ✅ Excelente. Resolver a colinearidade via SHAP agrupado é exatamente o que Lundberg recomenda. A soma dos SHAP por grupo é exata e aditiva — isso é matematicamente garantido pelo TreeSHAP. |
| **Migração de Crédito LSTM (Fig. 7)** | ✅ Argumento muito forte e visualmente claro. A linha preta (soma total de H2) é estável enquanto as barras (h14 vs h28) flutuam. Isso é a prova visual mais limpa de que as features LSTM são intercambiáveis. |
| **Abordagem da tese negativa do LSTM** | ✅ A coragem de refutar a própria hipótese é maturidade acadêmica. A banca valoriza isso enormemente. |

### 1.3 O que está ERRADO ou PERIGOSO

> [!CAUTION]
> **Problema 1: Os eixos X dos gráficos de dependência (Fig. 4) estão em unidades ambíguas para extrair limiares.**

Olhei o código em [fase6_shap_track_b.py:54-78](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/tcc-transformer-fault/src/evaluation/fase6_shap_track_b.py#L54-L78). O `unscale_features()` faz `inverse_transform` do MinMaxScaler e depois `expm1`. Mas olhe o gráfico `fig4_thresh_ratio_C2H4_TH_D1.png`: o eixo x vai de 0.0 a 0.9 — **isso é a escala bruta da razão (proporção, 0 a 1), não é percentual**. Quando o documento diz "o limiar foi aprendido em **23.4%**", isso precisa ser verificado: o valor 0.234 no gráfico corresponde a 23.4% **somente se** `ratio_C2H4_TH` foi definido como `C2H4 / TH` (que dá uma fração, não um percentual). Isso está correto, mas **o texto precisa ser explícito** sobre essa conversão, senão a banca pergunta "de onde veio o 23.4%?" e você gagueja.

**Já o gráfico de `ratio_C2H2_C2H4 -> D1`**: o eixo x vai de 0 a **730**. Isso são as razões brutas em ppm/ppm. O texto diz "o limiar é > 1.96", mas olhando o gráfico, os pontos com SHAP positivo estão concentrados abaixo de x ≈ 20, com o cruzamento do zero difícil de localizar visualmente. **O limiar de 1.96 não é visível no gráfico como está** — a escala está comprimida por outliers extremos (x = 730). Precisa de escala log ou filtro de outliers.

> [!CAUTION]
> **Problema 2: "SHAP = 0" ≠ fronteira de decisão.**

O documento trata "onde o SHAP cruza zero" como se fosse o limiar de decisão. **Não é.** SHAP = 0 significa "essa feature contribui tanto quanto a média da base para essa classe". A fronteira de decisão real depende da **soma de TODOS os SHAPs + bias**. Portanto, o "limiar empírico" extraído pelo cruzamento do zero é na verdade o ponto onde a **contribuição marginal daquela feature muda de sinal** — o que é válido e interpretável, mas precisa ser descrito com precisão.

**Como corrigir no texto:** Troque *"o limiar onde o impacto do SHAP se inverteu a favor da classe D1"* por *"o limiar onde a contribuição marginal de %C₂H₄ para o log-odds de D1 muda de negativa para positiva, indicando que, a partir dessa concentração, essa feature isoladamente empurra a predição na direção de D1"*.

> [!WARNING]
> **Problema 3: A afirmação "%C₂H₄ -> D1: match perfeito com 23% de Duval" é exagerada.**

Olhando o gráfico `fig4_thresh_ratio_C2H4_TH_D1.png`, o cruzamento do zero está **claramente entre 0.20 e 0.25** (ou seja, 20% a 25%). Chamar isso de "23.4%" com uma casa decimal implica uma precisão que o gráfico (scatter sem suavização) não sustenta. O que se pode dizer com honestidade é:

*"O limiar de inversão da contribuição marginal de %C₂H₄ para D1 encontra-se na faixa de 20–25%, alinhada com o limite de 23% do Triângulo de Duval."*

Isso é um achado forte o suficiente sem fingir precisão decimal.

> [!WARNING]
> **Problema 4: Faltam beeswarms das classes PD, T1, T2 e Normal.**

O documento faz afirmações sobre PD ("CH₄/H₂ domina"), T1 e T2 ("crédito dividido nas razões de Metano e Etileno"), mas os beeswarms gerados são apenas D1, D2 e T3. **As afirmações sobre PD, T1 e T2 não têm suporte gráfico direto.** A banca pode notar isso. O heatmap (Fig. 1) cobre parcialmente, mas um beeswarm de PD mostrando `ratio_CH4_H2` como dominante seria muito mais convincente.

---

## 2. Suficiência dos Arquivos Gerados

### 2.1 Tabela de cobertura: Afirmação -> Evidência

| # | Afirmação no Documento | Arquivo de Suporte | Suficiente? |
|:-:|---|---|:-:|
| 1 | Estabilidade ρ = 0.963 | `shap_stability.json` | ✅ |
| 2 | Features agrupadas por família resolvem a colinearidade | `fig2_grouped_track_b.png` | ✅ |
| 3 | Descargas dominadas por derivados de C₂H₂ | `fig1_heatmap_track_b.png` + `fig3_beeswarm_D1.png` | ✅ |
| 4 | D2 dominado por %CH₄/TH (baixo = descarga) | `fig3_beeswarm_D2.png` | ✅ |
| 5 | %C₂H₄ -> D1: limiar ~23% | `fig4_thresh_ratio_C2H4_TH_D1.png` | ⚠️ Precisa de suavização |
| 6 | %C₂H₂ -> D1: limiar ~8% | `fig4_thresh_ratio_C2H2_TH_D1.png` | ⚠️ Escala dificulta leitura precisa |
| 7 | C₂H₂/C₂H₄ -> D1: limiar > 1.96 | `fig4_thresh_ratio_C2H2_C2H4_D1.png` | ❌ Escala comprimida, limiar invisível |
| 8 | C₂H₄/C₂H₆ -> T3: limiar ~2.97 | `fig4_thresh_ratio_C2H4_C2H6_T3.png` | ⚠️ Escala comprimida |
| 9 | PD dominado por CH₄/H₂ | `fig1_heatmap_track_b.png` | ⚠️ Só heatmap, sem beeswarm |
| 10 | Faltas térmicas usam C₂H₄ e CH₄ | `fig3_beeswarm_T3.png` | ⚠️ Faltam T1 e T2 |
| 11 | LSTM não agrega (migração de crédito) | `fig7_lstm_credit_migration.png` | ✅ |
| 12 | Aditividade do SHAP verificada | — | ❌ **NÃO FOI VERIFICADA** |
| 13 | Os erros do modelo estão na fronteira de zona | `erros_test_seed42.csv` | ⚠️ Só CSV bruto, sem gráfico |

**Veredito: 5/13 plenamente suportados, 5 parciais, 3 com lacunas.**

### 2.2 Verificação Técnica do Código

Examinei [fase6_shap_track_b.py](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/tcc-transformer-fault/src/evaluation/fase6_shap_track_b.py) linha a linha:

| Aspecto | Status | Comentário |
|---|---|---|
| Uso de `pred_contribs=True` (TreeSHAP nativo) | ✅ | Correto, não passa pela biblioteca `shap` para calcular, só para plotar |
| Remoção do bias (última coluna) | ✅ | Linha 211: `contribs[:, :, :-1]` |
| Estabilidade com Spearman | ✅ | Mas inclui a diagonal (ρ=1 consigo mesmo), inflando a média. Deveria ser `np.mean(rho_matrix[np.triu_indices(5, k=1)])` |
| `pred_interactions=True` | ✅ | Usado para extrair main effects |
| Extração do main effect (diagonal) | ✅ | Linha 156: `interactions[:, c_idx, f_idx, f_idx]` |
| Desnormalização | ⚠️ | Funciona, mas `unscale_features` assume que features ausentes do subset C3 podem ser zeradas — o `inverse_transform` de um MinMaxScaler com zeros nas colunas ausentes pode distorcer se o `data_min_` não é zero. Precisa de teste unitário. |
| **Aditividade (`np.allclose`)** | ❌ | **NÃO FOI FEITA.** O `parecer_banca_fase6.md` mandou fazer (§4.2 item 3) e não foi implementado |

> [!IMPORTANT]
> **A falta da verificação de aditividade é o maior risco técnico.** Se `sum(shap_values) + bias ≈ margem` não for verificada, toda a análise pode estar corrompida silenciosamente. É uma linha de código. Acrescente.

---

## 3. Identificação de Lacunas e Sugestões

### 3.1 Lacunas CRÍTICAS (devem ser corrigidas)

| # | O que falta | Por quê | Esforço |
|:-:|---|---|:-:|
| 1 | **Verificação de aditividade** (`np.allclose(shap.sum() + bias, margin)`) | Valida que o TreeSHAP está funcionando corretamente. Sem isso, tudo é castelo de cartas | 10 min |
| 2 | **Escala log nos gráficos fig4 de razões** (C₂H₂/C₂H₄, C₂H₄/C₂H₆) | Outliers comprimem a escala e tornam o limiar invisível. Os gráficos atuais não sustentam os limiares citados | 30 min |
| 3 | **Suavização (LOWESS ou mediana por quantil)** nos gráficos fig4 + **anotação do cruzamento de zero** | O `parecer_banca_fase6.md` §4.4 mandou fazer. Sem isso, o "23.4%" é leitura visual, não cálculo | 1-2 h |
| 4 | **Correção do cálculo de estabilidade** para excluir a diagonal (cada seed vs si mesma = 1.0) | O ρ médio real é ligeiramente menor que 0.963. Provavelmente ainda > 0.90, mas precisa ser honesto | 5 min |

### 3.2 Lacunas IMPORTANTES (agregam valor significativo)

| # | O que gerar | O que agrega | Esforço |
|:-:|---|---|:-:|
| 5 | **Beeswarms de PD, T1, T2 e Normal** (todas as 7 classes) | Sustenta as afirmações sobre PD e classes térmicas. A banca pode perguntar "cadê o gráfico de PD?" | 15 min |
| 6 | **Waterfall plots de 2-3 erros típicos** (ex: D1->D2, Normal->T3) do CSV de erros | Mostra POR QUE o modelo errou em casos individuais. É a figura mais didática para a defesa oral | 1 h |
| 7 | **Triângulo de Duval do teste** com pontos coloridos por rótulo e erros do C3 destacados | A **Fig. 5 do plano original** — foi planejada mas não gerada. É provavelmente a figura mais forte para o TCC inteiro | 2-3 h |
| 8 | **Tabela formal de limiares** com IC (bootstrap x seeds) vs limiar normativo | O `parecer_banca_fase6.md` §4.4 planejou. Hoje os limiares são "lidos" do gráfico a olho | 2 h |

### 3.3 Lacunas OPCIONAIS (nice to have)

| # | O que gerar | O que agrega | Esforço |
|:-:|---|---|:-:|
| 9 | **SHAP interaction plots** (ex: C₂H₂_TH x C₂H₄_TH para D1) | Mostra como duas features interagem para segregar classes. Elegante, mas não essencial | 1-2 h |
| 10 | **Force plot de uma amostra "canônica"** de cada classe (a mais bem classificada) | Complementa o waterfall. Mostra como o modelo "pensa" sobre o caso ideal | 30 min |
| 11 | **Histograma de splits do booster** (`trees_to_dataframe()`) por feature em unidades físicas | A **Fig. 4 do plano original** — evidência independente dos cortes, não baseada em SHAP. Dupla confirmação | 1-2 h |

---

## 4. Tutoria Completa: Como Interpretar Cada Gráfico

### 4.1 Heatmap de Importância Global por Classe (Fig. 1)

**O que é:** Uma matriz Classes x Features. Cada célula mostra a **média de |SHAP|** (valor absoluto) para aquela feature naquela classe. Quanto mais escuro (vermelho), mais aquela feature influencia as decisões sobre aquela classe.

**Como ler na prática:**
1. **Linhas escuras em poucas colunas** = feature especialista. Exemplo: `ratio_CH4_H2` é escuríssimo em PD e D1, mas claro em T1/T2/T3. Isso significa que a razão CH₄/H₂ é um "switch" que o modelo usa quase exclusivamente para separar PD e descargas do resto.
2. **Linhas uniformemente claras** = feature com pouco impacto global. Exemplo: `CH4`, `C2H6` são claras em quase tudo — o modelo pouco as usa diretamente (o crédito já está nas razões derivadas).
3. **Colunas uniformemente escuras** = classe "difícil" que requer muitas features. Olhe a coluna "Normal" — ela é escura em H2, TH1 e na família de gases puros. Normal é a "classe de exclusão" (nenhum gás alto).

**O que a banca vai olhar:**
- *"As descargas (D1/D2) usam features de acetileno? As térmicas usam etileno?"* Se sim, o modelo aprendeu a física. Se não, suspeita de overfitting.
- **Seu heatmap confirma:** D1/D2 dominadas por `ratio_C2H2_TH1`, `ratio_C2H2_C2H6`, `ratio_C2H2_TH`. T3 dominada por `ratio_C2H6_TH1`, `ratio_C2H4_C2H6`. PD dominada por `ratio_CH4_H2`. ✅ Tudo coerente com a física.

---

### 4.2 Barplot Agrupado por Família (Fig. 2)

**O que é:** As features individuais foram somadas em 4 "famílias" (Gases puros, Razões IEC/Duval, Família TH, Família TH1). A soma do SHAP agrupado é **exata** e **aditiva** — é a propriedade matemática de Shapley. Portanto, não se perde informação ao somar.

**Como ler:**
1. **Barra mais alta** = classe que "exige mais explicação" do modelo (mais distante da predição média). Normal e PD têm barras altas porque são extremos (Normal = nada anormal, PD = assinatura muito específica de H₂).
2. **Cor dominante em cada barra** = família que mais contribui para diagnosticar aquela classe.
   - D1/D2: dominadas por **Razões IEC/Duval** (azul) e **Família TH1** (amarelo). Descargas são diagnosticadas por razões de acetileno, que aparecem tanto nas razões IEC (C₂H₂/C₂H₄) quanto nos percentuais de Duval (C₂H₂/TH1).
   - T1/T2/T3: contribuição mais distribuída, com **Família TH1** significativa. As térmicas dependem de proporções relativas (percentuais), não de uma única razão.
3. **Gases puros (roxo escuro) pouco contribuem em D1/D2**. Isso mostra que o modelo aprendeu que a concentração absoluta importa menos que a proporção para descargas — exatamente o que Duval e a IEC dizem.

**O que a banca vai olhar:**
- *"Por que você agrupou?"* Resposta: porque features derivadas do mesmo gás (ex: C₂H₂, C₂H₂/TH, C₂H₂/TH1, C₂H₂/C₂H₄) competem pelo crédito no SHAP individual. Sem agrupamento, cada uma parece pouco importante. Agrupadas, o sinal real aparece.

---

### 4.3 Beeswarm Plot (Fig. 3: D1, D2, T3)

**O que é:** O gráfico mais informativo do SHAP. Cada ponto é uma amostra do teste. O eixo x é o valor de SHAP (contribuição para o log-odds daquela classe). A cor indica o valor real da feature naquela amostra (azul = baixo, vermelho = alto).

**Como ler, passo a passo:**

1. **Features no topo** = maior impacto médio (ordenadas por média de |SHAP|).
2. **Pontos à direita do zero** = essa feature EMPURRA a predição PARA essa classe.
3. **Pontos à esquerda** = essa feature EMPURRA a predição PARA FORA dessa classe.
4. **A COR revela a relação feature-classe:**
   - Se pontos **vermelhos (valor alto)** estão à **direita** → valores altos favorecem a classe. Exemplo: No beeswarm de D1, `ratio_C2H2_TH1` vermelho (alto acetileno) está à direita → **alto acetileno favorece D1**. ✅ Correto fisicamente.
   - Se pontos **azuis (valor baixo)** estão à **direita** → valores baixos favorecem a classe.
   - No beeswarm de D2, `ratio_CH4_TH` azul (pouco metano relativo) está à esquerda com magnitude alta → D2 tem metano relativamente baixo em proporção (o acetileno domina).
5. **Espalhamento horizontal** = dispersão do impacto. Features com pontos espalhados são "decisoras condicionais" — importam muito em certas amostras e pouco em outras.

**O que a banca vai perguntar:**
- *"No beeswarm de T3, por que `ratio_C2H6_TH1` é a feature mais importante?"* → Porque a T3 desta base tem química atípica (mediana de C₂H₄/C₂H₆ = 0.25, quando a IEC espera > 4). O modelo aprendeu que alto %C₂H₆ (etano) relativo é a assinatura de T3 **nesta base**, não da norma. Cite isso como **achado sobre os dados**.

**Lacuna que impacta a defesa:** Você tem beeswarms de D1, D2 e T3, mas **não de PD**. No heatmap, PD mostra `ratio_CH4_H2` com o maior SHAP de todo o projeto (~1.3). Um beeswarm de PD mostrando essa dominância seria a figura mais bonita e convincente do TCC.

---

### 4.4 Gráficos de Dependência / Limiares (Fig. 4)

**O que é:** Cada gráfico mostra o **Main Effect** (efeito principal, extraído da diagonal da matriz de interações SHAP) de uma feature para uma classe. Eixo x = valor da feature (desnormalizado). Eixo y = contribuição SHAP isolada dessa feature. A linha tracejada em y = 0 marca a média.

**Como ler:**

1. **Padrão sigmoidal / degrau** = existe um limiar claro. Exemplo perfeito: `ratio_C2H4_TH -> D1`. Abaixo de ~0.20 (20%), SHAP positivo (favorece D1). Acima de ~0.25, SHAP fortemente negativo (desfavorece D1). **Isso é o modelo dizendo: "D1 tem pouco etileno relativo (< 23%), e eu aprendi isso sozinho."**

2. **Padrão monotônico** = sem limiar discreto, mas uma tendência. Exemplo: `ratio_C2H2_TH -> D1`. SHAP sobe continuamente com %C₂H₂, cruzando zero em torno de 5–10%. **Mais acetileno = mais chance de D1**, sem um corte abrupto.

3. **Dois "patamares" separados por um salto** = o modelo tem uma regra binária forte. É isso que se vê em `ratio_C2H4_TH -> D1`: dois patamares claros (+0.17 e -0.30) com transição em ~0.22.

**ATENÇÃO aos seus gráficos atuais:**

- `ratio_C2H2_C2H4 -> D1`: A escala x vai até 730 porque há outliers extremos. **Todo o sinal está comprimido em x < 20.** Refaça com `plt.xlim(0, 30)` ou escala log. Sem isso, o limiar de 1.96 que você cita é impossível de verificar visualmente.
- `ratio_C2H4_C2H6 -> T3`: Mesma compressão (x até 650). Os dados informativos estão todos em x < 30.
- `ratio_C2H4_TH -> D1`: Este é o melhor gráfico. A transição é visualmente clara. **Mas adicione uma linha de suavização (LOWESS) e uma anotação com seta indicando o ponto de cruzamento.** Sem isso, o leitor não sabe onde exatamente está o 23%.

**O que a banca vai perguntar:**
- *"De onde veio o número 23.4%?"* Se você não tem uma suavização formal, diga: "o limiar se encontra na faixa de 20–25%, consistente com os 23% do Triângulo de Duval". Não cite uma casa decimal sem LOWESS + bootstrap.
- *"SHAP = 0 é a fronteira de decisão?"* **NÃO.** SHAP = 0 é onde a contribuição marginal daquela feature é neutra. A decisão depende da soma de todas. Explique isso proativamente.

---

### 4.5 Migração de Crédito LSTM (Fig. 7)

**O que é:** Um gráfico que mostra como o SHAP de H₂ se distribui entre `lstm_h14_H2` e `lstm_h28_H2` em cada uma das 5 seeds do XGBoost. A linha preta mostra a **soma total** (impacto total de H₂ como conceito).

**Como ler:**
1. **As barras variam entre seeds** (~65% h14 / ~35% h28 na seed 42, mudando para ~68/32, ~67/33 etc). Essa variação é **estocástica** — artefato da inicialização aleatória das árvores.
2. **A linha preta é plana** (varia de ~0.47 a ~0.49). A **soma é estável**, só a distribuição interna flutua.

**O que isso prova:**
- As features `lstm_h14_H2` e `lstm_h28_H2` são **intercambiáveis** do ponto de vista do modelo. O XGBoost não "prefere" uma sobre a outra — ele usa whichever happens to appear first nos splits daquela seed.
- Portanto, o LSTM **não adicionou informação nova**. Se tivesse, `lstm_h28_H2` consistentemente carregaria um crédito extra estável (representando a informação de tendência de longo prazo) que `lstm_h14_H2` não captura.

**O que a banca vai perguntar:**
- *"Isso prova que o LSTM é inútil em geral ou só nesta base?"* Resposta: **nesta base e para este alvo (diagnóstico do estado atual)**. Para prognóstico (prever falha futura), o LSTM poderia ter valor — mas esse não é o objetivo deste TCC.
- *"Por que a soma não é perfeitamente plana?"* Porque cada seed gera árvores ligeiramente diferentes, e o conjunto de teste é finito (295 amostras). A variação de ~0.02 é ruído amostral.

> [!TIP]
> **Dica para a defesa:** Este gráfico deve ser o último slide antes da conclusão sobre o LSTM. Ele é autoexplicativo e fecha o argumento sem apelação. Não precisa de fórmulas — a estabilidade da linha preta vs. a instabilidade das barras conta a história inteira.

---

### 4.6 O Heatmap vs. Beeswarm: Quando Usar Cada

| Característica | Heatmap (Fig. 1) | Beeswarm (Fig. 3) |
|---|---|---|
| **Visão** | Panorâmica (todas as features x todas as classes) | Detalhada (todas as features, UMA classe) |
| **Informação** | Magnitude do impacto | Magnitude + direção + distribuição + relação com o valor da feature |
| **Quando usar** | Para mostrar o "mapa geral" do modelo | Para argumentar sobre a física de uma classe específica |
| **Limitação** | Não mostra direção (sempre \|SHAP\|) | Precisa de um por classe (7 classes = 7 figuras) |

**Recomendação:** No TCC, use o heatmap como visão geral e 3–4 beeswarms selecionados (D1, D2, T3 e PD) com narrativa física. Não coloque 7 beeswarms no corpo do texto — os extras vão para o apêndice.

---

## 5. Checklist de Ações Prioritárias

### Prioridade ALTA (antes de escrever o TCC)

- [ ] **Adicionar verificação de aditividade** ao script. Uma linha: `assert np.allclose(shap_values.sum(axis=-1) + bias, margin, atol=1e-5)`
- [ ] **Corrigir o cálculo de estabilidade** para usar apenas os pares off-diagonal: `rho_matrix[np.triu_indices(5, k=1)]`
- [ ] **Refazer gráficos fig4 de razões** com escala log ou xlim restrito + suavização LOWESS + anotação do cruzamento
- [ ] **Gerar beeswarms de PD, T1, T2 e Normal** (são 4 linhas a mais no loop do script)
- [ ] **Gerar 2-3 waterfall plots** dos erros mais representativos do CSV

### Prioridade MÉDIA (para fortalecer)

- [ ] **Gerar o Triângulo de Duval** com pontos do teste coloridos e erros destacados
- [ ] **Formalizar a tabela de limiares** com IC via bootstrap
- [ ] **Gerar histograma de splits** do booster por feature

### Prioridade BAIXA (se sobrar tempo)

- [ ] SHAP interaction plots
- [ ] Force plots de amostras canônicas

---

## 6. Resumo Final

| Aspecto | Nota | Comentário |
|---|---|---|
| **Metodologia SHAP** | 8/10 | TreeSHAP correto, main effects, estabilidade. Falta aditividade. |
| **Gráficos gerados** | 6/10 | Boa cobertura de D1/D2/T3, mas escalas ruins em fig4 e classes faltantes nos beeswarms |
| **Documento de interpretação** | 7/10 | Narrativa inteligente e madura, mas 2 afirmações exageradas e 1 confusão SHAP=0 vs. fronteira |
| **Argumento LSTM** | 9/10 | Migração de crédito é brilhante. Gráfico limpo e conclusão honesta. |
| **Prontidão para defesa** | 6/10 | Precisa das correções e dos beeswarms extras para não tropeçar em pergunta sobre PD/T1 |

> [!IMPORTANT]
> **O trabalho é bom. O rigor está acima da média de um TCC.** As lacunas apontadas são refinamentos que separam um trabalho "aprovado" de um trabalho "aprovado com distinção". O argumento central — o modelo aprendeu a física dos gases — é sustentável com os dados, desde que você não exagere a precisão dos limiares e corrija as escalas dos gráficos.
