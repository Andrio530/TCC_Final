# Parecer de Banca: Fase 5.5 e Plano da Fase 6

**Data:** 06/10 · **Entrega:** 15/11
**Base do parecer:** verificação direta de `fase55_track_a.json`, `fase55_track_b.json`, `classification_test.csv` (ppm brutos), `china_features_*.csv` e do código de features e baselines. Não me baseei só no relatório.

> [!IMPORTANT]
> **Veredito:** o trabalho é honesto e metodologicamente limpo (sem leakage, com ablação e IC). **Mas três afirmações do relatório atual não se sustentam nos dados**, e uma delas fui eu que introduzi. Elas precisam ser corrigidas antes da escrita, senão a banca derruba com uma pergunta. Pode seguir para a Fase 6 depois de **~2 dias de verificações (Fase 5.6)**.

---

## 1. Correções de afirmações (o que está errado hoje)

### 1.1 "O LSTM entrega uma representação mais limpa/suavizada que ajuda as árvores": FALSO
Fui eu que escrevi isso na resposta anterior, e os dados não sustentam.

| Evidência | Valor |
|---|---|
| Correlação `lstm_h14_X` × `obs_X` (treino) | **r = 0,9993 a 0,9997** |
| Correlação `lstm_h28_X` × `obs_X` | **r = 0,9974 a 0,9990** |
| C6 (só LSTM) × C5a (só observados), XGBoost default | 0,889 × 0,886 (Δ ≈ +0,003, dentro do ruído) |

Se a "versão suavizada" ajudasse, C6 seria melhor que C5a. Não é. **As features LSTM são praticamente cópias de `obs_X`.** O IGA escolher `lstm_h28_H2` em vez de `obs_H2` é uma **escolha arbitrária entre gêmeas**, com fitness praticamente igual, e não uma preferência com significado.

> [!WARNING]
> Não cite `C1 − C1_menos_LSTM_posthoc = +0,063 (IC exclui zero)` como prova de valor do LSTM. Esse número cai porque, ao retirar as `lstm_*` do C1, o modelo **perde H₂, C₂H₄ e C₂H₂ por completo**: o IGA tinha escolhido as gêmeas LSTM no lugar das observadas. É um artefato da ablação post-hoc. O teste justo é C1 × C5b.

Também troque "p > 0,05" por "**IC95% da diferença pareada contém zero**", que é o que foi de fato calculado.

### 1.2 "Duval é ruim por causa da taxa de não-classificação": FALSO para Duval

| Método | Cobertura (falhas) | Acurácia entre as classificadas |
|---|:---:|:---:|
| Duval | **90,5%** | **53,8%** |
| IEC 60599 (razões) | 60,3% | 42,1% |

A não-classificação explica parte da IEC, mas **não explica o Duval**: ele classifica 9 em cada 10 amostras e erra metade delas. O motivo real está no item 1.3. (Detalhe: o F1 de Duval **subiu** de 0,26 para 0,36 com a correção, não caiu.)

### 1.3 "Os rótulos foram atribuídos com os mesmos critérios de gás → confusão de fronteira": CONTRADITO para as classes térmicas
Medianas no teste, em ppm brutos:

| Classe | C₂H₄/C₂H₆ (IEC: T1<1, T2 1–4, **T3>4**) | %C₂H₄ Duval (T1<20, T2 20–50, **T3>50**) | Coerente com a norma? |
|---|:---:|:---:|:---:|
| T1 | 0,94 | **30,6** | ⚠️ cai em T2 no Duval |
| T2 | 1,84 | 42,6 | ✅ |
| **T3** | **0,25** | **31,6** | ❌ química de T1/T2, não de >700 °C |
| D1 | 0,39 | 10,7 (%C₂H₂ = 55) | ✅ |
| PD | CH₄/H₂ = 0,07 | %CH₄ ≈ 97 (limite = 98) | ✅ IEC, ⚠️ limítrofe no Duval |

As matrizes confirmam isso: **IEC manda 36/66 T3 para T1** e **Duval manda 28/66 T3 para T2**. Em DGA, T3 (>700 °C) gera muito etileno (C₂H₄ ≫ C₂H₆). Nesta base, a "T3" típica tem mais etano que etileno.

**Interpretação honesta:** a base é uma **compilação multi-fonte** (IEEE DataPort + literatura). Os rótulos térmicos provavelmente vêm de critérios heterogêneos (inspeção, normas chinesas, outros autores) e **não seguem a física da IEC 60599**. Consequências:
- O baseline normativo perde sobretudo nas térmicas e em PD. **Isso não é bug de implementação**, é discordância entre rótulo e norma.
- O ML vai a ~0,80 porque **aprende a convenção de rotulagem desta base, incluindo o ruído dela**. A comparação ML × norma favorece o ML por construção, e isso precisa estar escrito.
- **Isso afeta a Fase 6 diretamente:** nas classes térmicas, os limiares do SHAP **não vão coincidir** com os da IEC. Isso deve ser previsto *antes* de rodar (ver §4.4).

---

## 2. Respostas às suas três perguntas

### 2.1 A teoria do LSTM (falhas lentas → horizonte curto não muda nada) é defensável?
**Parcialmente. A intuição física é boa, mas não é a explicação principal, e a extensão ("ir para meses") não resolve.**

- O argumento decisivo é **informacional, não químico**. A previsão LSTM é uma função determinística da janela passada (50 pontos × 12 h ≈ 25 dias). Prever mais longe **não cria informação**, só extrapola a mesma janela com erro maior (seu `multi_horizon_eval.py` já mostra o R² caindo com h). Para diagnóstico, o que poderia ajudar é a **tendência contida no histórico**, e não o horizonte.
- Na Katser, o rótulo é do **estado atual** da série e o snapshot `obs_t` já separa as classes (F1 ≈ 0,886 sem LSTM). Não sobra muita coisa para a tendência explicar.
- "Ir para meses" só faz sentido se o **alvo** mudar para um rótulo futuro (prognóstico, ou seja, transição Normal → falha). Confirme se a Katser tem transições dentro das séries ou rótulo constante por série. Se for constante, **não há transição para prever** e prognóstico de classe não é possível com essa rotulagem. Isso vai para Trabalhos Futuros, sem prometer nada.
- Detalhe que a banca pode pegar: 3.000 séries de 420 pontos com amostragem regular de 12 h é atípico para DGA laboratorial (normalmente mensal ou trimestral). Confirme no artigo de Katser (2022) se as séries são **medições de sensores online, simuladas ou aumentadas**, e escreva isso explicitamente. Se forem sintéticas, o argumento físico sobre T1 perde força.

**Como escrever:** *"A hipótese de que projeções temporais agregariam poder diagnóstico não se confirmou: as projeções têm r > 0,997 com as concentrações atuais e são intercambiáveis com elas (C6 ≈ C5a). Para o diagnóstico do estado presente, a informação está no snapshot; a utilidade de modelos temporais deve ser investigada em formulações de prognóstico."* Isso é um resultado negativo bem fundamentado, não uma desculpa. **Desculpa seria** dizer que o LSTM "suaviza ruído e por isso o IGA o prefere", o que os dados contradizem.

### 2.2 Dizer "equivalente a Wu et al." fortalece ou enfraquece?
**Fortalece a honestidade, mas expõe a pergunta mais perigosa da defesa:**

| Comparação (Δ F1, IC95% pareado) | Resultado |
|---|---|
| IGA (C3) × default 19 features (C4), Track B | +0,013 [−0,021; +0,049] |
| IGA (C1) × default (C2), Track A | 0,000 [−0,014; +0,012] |
| IGA sem LSTM (C5b) × default sem LSTM (C5a) | −0,006 [−0,025; +0,012] |
| Wu (T3) × default (C4) | −0,001 [−0,041; +0,038] |

> [!CAUTION]
> **"Por que não usar só o XGBoost default com todas as features?"** Hoje não há resposta baseada em desempenho. A "parcimônia" também é fraca: 19 → 16 no Track B (−16%). No Track A, 17 → 8 inclui remover 8 gêmeas redundantes; sem LSTM, o IGA vai de 9 → 8. E o IGA custa ~1,6 h por execução contra segundos do default.

**Reenquadramento defensável da contribuição** (honesto e suficiente para um TCC):
1. **Estudo comparativo rigoroso** (ablação + IC pareado) que mostra que, em DGA com 5 gases, nem seleção evolutiva, nem features da literatura (Wu), nem projeções LSTM superam um XGBoost bem regularizado: **um resultado negativo útil para a área**, que costuma reportar ganhos sem IC.
2. **Auditoria de rótulos × norma** (§1.3): mostra por que baselines normativos "falham" em bases compiladas. Isso é contribuição real.
3. **Interpretabilidade (Fase 6):** onde o modelo concorda e onde discorda da IEC/Duval.

Escreva "reimplementamos o conjunto de features de Wu et al. na mesma partição". Não escreva "reproduzimos Wu et al.", a menos que compare com o número publicado por eles.

### 2.3 A crítica a Duval/IEC justifica o ML?
**Com a narrativa atual, não.** A narrativa defensável é: *"Na base compilada, os métodos normativos concordam com os rótulos em ~50% das falhas classificadas, com discordância concentrada em T3, T1 e PD, onde a assinatura química rotulada difere da prevista pela norma. O ML atinge ~0,80 porque aprende a convenção da base, com cobertura total."* Isso justifica o ML como **ferramenta adaptável a um padrão de rotulagem**, e não como "melhor que a física".

Para blindar a implementação, crie um **teste unitário** com pontos canônicos de cada zona do Triângulo 1 (vértices e centros de zona publicados por Duval e na IEC 60599 Anexo B) e das razões da Tabela 1 da IEC. É 1 hora de trabalho e elimina a pergunta "sua implementação está certa?".

---

## 3. Fase 5.6: verificações obrigatórias antes do freeze (07–08/10)

Não são features novas: são **testes de hipóteses já afirmadas no texto**.

| # | Tarefa | Afirmação que sustenta | Esforço |
|:-:|---|---|:-:|
| 1 | **Baseline só com gases brutos** (Track B: 5 gases em ppm; Track A: 4 `obs_*`), XGBoost default, mesmo bootstrap pareado | "Teto informacional dos 5 gases" (hoje é só afirmação). Se der ≈0,78, prova que a engenharia de features também não agrega | 1 h |
| 2 | **Tabela de concordância rótulo × norma** por classe (% de amostras de cada rótulo na zona Duval/IEC correspondente), em treino+val+teste | §1.3. Explica o baseline e prevê o SHAP | 2 h |
| 3 | **Testes unitários** de Duval e IEC com pontos canônicos | Credibilidade dos baselines | 1 h |
| 4 | **Repeated Stratified K-Fold 5×5** (treino+val) para C3, T3, C4 e o baseline bruto, hiperparâmetros fixos | Val 0,704 × teste 0,797 é uma diferença grande (a sobreposição de ICs não é um teste). Dá uma estimativa única e estável | 3–4 h |
| 5 | *(Opcional)* XGBoost default + **features de tendência** (Δ e inclinação de `obs_X` na janela de 50 pontos) no Track A | Separa "o LSTM é inútil" de "a tendência é inútil" e fecha a questão do horizonte. Decisão pré-registrada: se o IC contém zero, a tendência não agrega para diagnóstico | 3 h |

**Feature freeze: 09/10.** Nenhum resultado desses itens reabre a busca por features.

---

## 4. Plano da Fase 6: SHAP e validação físico-química (10–17/10)

### 4.1 Escopo
- **Principal: Track B, modelo C3** (tem os 5 gases → dá para mapear o triângulo de Duval e as razões IEC).
- **Secundário: Track A**. Sem CH₄ e C₂H₆, não há Duval nem IEC completa; serve só para a narrativa do LSTM e uma leitura física curta. Não gaste mais de 1 dia nele.
- Objetivo: **não mexer no F1.** Explicar, confrontar com a norma e documentar concordâncias e divergências.

### 4.2 Armadilhas técnicas (leia antes de codar)
1. **As features estão em MinMax** (`china_feature_scaler.pkl`, `katser_feature_scaler.pkl`). O eixo x de todo gráfico de dependência **precisa ser invertido para unidades físicas** (ppm, razão, %) antes de comparar com qualquer limiar normativo. As árvores não mudam (transformação monótona), mas os limiares em escala [0,1] não têm significado.
2. **Razões com denominador zero viram 0** (`np.where(d > 0, n/d, 0)`). Haverá um aglomerado artificial em x = 0, por exemplo C₂H₂/C₂H₄ com C₂H₄ = 0. Exclua ou marque essas amostras nos gráficos de dependência.
3. **Ambiente:** Python 3.14 + XGBoost 3.4 + shap 0.52. Use como fonte primária o **TreeSHAP nativo do XGBoost**: `booster.predict(DMatrix(X), pred_contribs=True)`, com saída (n, K, f+1), e `pred_interactions=True`. Use o `shap` só para plotar. Valide a **aditividade**: soma das contribuições + bias = margem, com `np.allclose`.
4. Saída multiclasse em **log-odds (margem)**, por classe. "SHAP = 0" significa "contribuição igual à média da base", **não** fronteira de decisão.
5. **Features correlacionadas** (TH, TH1, X/TH, X/TH1) dividem o crédito entre si. Por isso, o SHAP agrupado (soma por grupo, que é exata e aditiva) é obrigatório.

### 4.3 Passo a passo

| Passo | O que fazer | Saída |
|:-:|---|---|
| 1 | Carregar C3 (seed 42) e as 5 seeds; calcular `pred_contribs` e `pred_interactions` no **teste** (e no treino, para estabilidade); checar a aditividade | `shap_values_trackB.npz` |
| 2 | **Estabilidade:** Spearman do ranking de média de \|φ\| entre as 5 seeds | Tabela (se ρ < 0,8, não interprete a ordem fina) |
| 3 | **Importância por classe:** heatmap features × 7 classes de média de \|φ_k\| | **Fig. 1** (substitui 7 beeswarms) |
| 4 | **SHAP agrupado:** Brutos / Razões IEC (CH₄/H₂, C₂H₂/C₂H₄, C₂H₄/C₂H₆) / Outras razões / Família TH (coord. Duval) / Família TH1 | **Fig. 2** (barras empilhadas por classe) |
| 5 | Beeswarm para 3 classes-chave: D1, D2 e T3 | **Fig. 3** |
| 6 | **Extração de limiares** (§4.4) para os pares pré-registrados | **Tabela A** (limiar SHAP com IC × limiar normativo) |
| 7 | **Limiares de split do booster:** `trees_to_dataframe()`, histograma ponderado por ganho dos valores de split (em unidades físicas) por feature | **Fig. 4**: segunda evidência independente dos cortes |
| 8 | **Triângulo de Duval do teste:** pontos coloridos pelo rótulo, erros do C3 destacados | **Fig. 5**: a figura mais forte do TCC |
| 9 | **Análise de erro D1↔D2:** distância (em coordenadas Duval) até a fronteira D1/D2, erros × acertos, Mann-Whitney; waterfall de 2–3 erros típicos | **Fig. 6** + teste |
| 10 | **Track A:** SHAP agrupado por gás (`obs_X + lstm_h14_X + lstm_h28_X`) nas 5 seeds, mostrando que o crédito "migra" entre as gêmeas conforme a seed | **Fig. 7**: fecha a narrativa do LSTM |

### 4.4 Cruzamento SHAP × norma (núcleo acadêmico)

**Método:**
- Use a **dependência de efeito principal** (diagonal de `pred_interactions`, φ_ii), que remove a contaminação por interações e deixa o limiar mais limpo.
- Para cada par (feature, classe k): converta x para unidade física, aplique escala log para razões, suavize φ_ii,k(x) com medianas por quantil (ou LOWESS) e localize o cruzamento de zero.
- Obtenha o **IC do limiar** por bootstrap das amostras × 5 seeds.
- Compare com o limiar normativo e com o pico do histograma de splits (passo 7).

**Pré-registre as previsões antes de rodar** (isso transforma "olhar gráficos" em teste de hipótese):

| Feature (unid. física) | Classe | Limiar normativo | Previsão (pela §1.3) |
|---|---|---|---|
| %C₂H₂ = `ratio_C2H2_TH` | D1, D2 | 4% / 13% (Duval) | **Concorda** |
| %C₂H₄ = `ratio_C2H4_TH` | D1 × D2 | 23% (Duval) | **Concorda** (fronteira onde estão os erros) |
| C₂H₂/C₂H₄ | D1 | > 1 (IEC) | Concorda |
| CH₄/H₂ | PD | < 0,1 (IEC) | Concorda |
| %CH₄ = `ratio_CH4_TH` | PD | ≥ 98% (Duval) | Limiar aprendido **abaixo de 98** (PD da base está em ~97%) |
| C₂H₄/C₂H₆ | T3 | > 4 (IEC) | **Diverge** (T3 da base tem mediana 0,25) |
| %C₂H₄ | T1 × T2 × T3 | 20% / 50% (Duval) | **Diverge** em T1 e T3 |

**Leitura esperada para a escrita:** *o modelo recupera os limiares normativos onde rótulo e norma concordam (descargas, PD via IEC) e se afasta deles onde a rotulagem da base diverge da química esperada (T3, T1).* Isso é uma conclusão forte e honesta. **Divergência não é falha do modelo, é um achado sobre os dados.**

### 4.5 Entregáveis para o texto
- 7 figuras (Fig. 1–7), Tabela A (limiares) e uma tabela de estabilidade.
- Um parágrafo por classe ligando SHAP → gás → mecanismo físico (ex.: C₂H₂ ↔ arco/descarga; C₂H₄ ↔ temperatura do óleo; H₂ ↔ PD/corona).
- Seção de limitações: rótulos multi-fonte, natureza das séries Katser, SHAP ≠ causalidade, crédito dividido entre features correlacionadas.

---

## 5. Cronograma revisado

| Período | Atividade |
|---|---|
| 07–08/10 | Fase 5.6 (itens 1–4; 5 opcional) |
| 09/10 | Feature freeze + atualizar as tabelas do relatório |
| 10–14/10 | Fase 6, passos 1–7 (Track B) |
| 15–16/10 | Fase 6, passos 8–10 (Duval, erros, Track A) |
| 17/10 | Consolidar figuras e Tabela A |
| 18/10 → | Escrita (segue o cronograma original) |

---

## 6. Frases para a defesa

| ❌ Evitar | ✅ Usar |
|---|---|
| "O LSTM suaviza ruído e por isso o IGA o escolhe" | "As projeções LSTM são intercambiáveis com as concentrações atuais (r > 0,997; C6 ≈ C5a)" |
| "Duval falha por não classificar" | "Duval classifica 90% das falhas, mas concorda com o rótulo em 54%; a discordância está em T3, T1 e PD" |
| "Os erros estão na fronteira porque os rótulos seguem Duval" | "Nas descargas, os erros estão na fronteira D1/D2 (teste de distância); nas térmicas, os rótulos divergem da química normativa" |
| "O IGA reduz 53% das features" | "O IGA não superou o XGBoost default em nenhum cenário; o estudo mostra o teto de desempenho com 5 gases" |
| "Reproduzimos Wu et al." | "Reimplementamos as features de Wu et al. na mesma partição: equivalência estatística" |
