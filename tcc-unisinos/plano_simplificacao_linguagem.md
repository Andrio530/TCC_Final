# Plano Separado — Simplificação de Termos Complexos e Remoção de "Dois Pontos Explicativos" (Prioridade 3)

Este documento mapeia **todos os trechos da monografia (`tcc-unisinos/*.tex`)** divididos em duas frentes de limpeza de estilo:
- **Parte A (Itens P3.1 a P3.12):** Simplificação de termos excessivamente técnicos ou rebuscados ("verniz de IA") e de expressões como *"dinâmica temporal"*, sem usar dois pontos (`:`) nem travessões (`---`) para explicar conceitos.
- **Parte B (Itens P3.13 a P3.22):** Remoção sistemática das **inclusões de dois pontos (`:`) no meio das frases** para introduzir explicações (típico padrão sintático de IA), substituindo-as por conectivos naturais em português (*pois*, *visto que*, *uma vez que*, *em que* ou ponto final).

> **Nota:** Nenhuma dessas alterações foi aplicada ainda aos arquivos `.tex`. Você pode aprovar todas de uma vez ou indicar quais itens prefere aplicar.

---

## Parte A — Simplificação de Jargões e Termos Técnicos Complexos

### Item P3.1 — "Bifurcação central", "abordagem normativa reativa" e uso de "dinâmica" na Introdução
- **Local:** [`introducao.tex` (linhas 8–14 e linha 28)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/introducao.tex#L8-L28)
- **Como está hoje:**
  - Linhas 8–14: *"A figura também evidencia a bifurcação central que motiva este trabalho. Enquanto os métodos tradicionais realizam o diagnóstico de forma reativa, a abordagem proposta busca antecipar a ocorrência de falhas por meio de modelagem preditiva."*
  - Linha 28: *"Apesar do avanço desses modelos na literatura, a maioria dessas abordagens opera sobre snapshots estáticos dos dados, ignorando a dinâmica temporal da degradação."*
- **O que quer dizer na prática:** As normas (IEC/Duval) só avaliam o equipamento depois que o gás já subiu; já os modelos preditivos tentam estimar a evolução dos gases ao longo do tempo e automatizar a classificação.
- **Sugestão mais simples e natural (sem dois pontos):**
  - Linhas 8–14: *"A figura também destaca a diferença prática que motiva este trabalho. Enquanto os métodos normativos tradicionais avaliam o equipamento apenas após a elevação dos gases, a abordagem com aprendizado de máquina busca projetar a evolução das concentrações e automatizar a identificação de falhas."*
  - Linha 28: *"Apesar do avanço desses modelos na literatura, a maioria dessas abordagens analisa apenas leituras isoladas no tempo, desconsiderando a evolução histórica da degradação."*

### Item P3.2 — Termos na Tabela 1 ("premissas distributivas", "expansão foliar", "problemas multimodais")
- **Local:** [`fundamentacao.tex` (linhas 141, 154 e 175)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L141-L175)
- **Como está hoje:**
  - Linha 141 (KNN): *"Simplicidade conceitual sem premissas distributivas"*
  - Linha 154 (LightGBM): *"Eficiência em bases volumosas por expansão foliar"*
  - Linha 175 (AG padrão): *"Risco de convergência prematura em problemas multimodais"*
- **O que quer dizer na prática:**
  - *Sem premissas distributivas:* o KNN não exige que os dados sigam uma distribuição estatística específica (como uma curva normal).
  - *Expansão foliar (leaf-wise):* o LightGBM cresce a árvore escolhendo a folha que dá maior queda de erro, o que causa *overfitting* quando há poucos dados.
  - *Problemas multimodais:* funções que têm vários ótimos locais, fazendo o algoritmo genético estagnar cedo.
- **Sugestão mais simples e natural:**
  - Linha 141 (KNN): *"Simplicidade de implementação e independência da distribuição dos dados"*
  - Linha 154 (LightGBM): *"Treinamento rápido em grandes bases por crescimento focado nas folhas"*
  - Linha 175 (AG padrão): *"Risco de estagnar cedo em ótimos locais"*

### Item P3.3 — "Expansão de Taylor em segunda ordem" no XGBoost
- **Local:** [`fundamentacao.tex` (linha 215)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L215)
- **Como está hoje:**
  > *"A expansão de Taylor em segunda ordem da função de perda permite calcular de forma analítica o ganho de divisão dos nós e os pesos ótimos das folhas, garantindo convergência eficiente e controle contra o sobreajuste."*
- **O que quer dizer na prática:** O XGBoost usa a primeira derivada (inclinação do erro) e a segunda derivada (curvatura do erro) para saber rapidamente qual ponto de corte nas árvores reduz mais o erro.
- **Sugestão mais simples e natural:**
  > *"O uso das derivadas de primeira e segunda ordem da função de perda permite calcular diretamente o ganho de cada divisão nos nós e o peso ideal das folhas, acelerando o treinamento e ajudando a evitar o sobreajuste."*

### Item P3.4 — "Produto elemento a elemento de Hadamard" e "dinâmicas temporais complexas" na LSTM
- **Local:** [`fundamentacao.tex` (linhas 255 e 265)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L255-L265)
- **Como está hoje:**
  - Linha 255: *"...$\odot$ representa o produto elemento a elemento de Hadamard, $f_t$ é a porta de esquecimento que modula o descarte de informações..."*
  - Linha 265: *"Para modelar dinâmicas temporais complexas em séries temporais de gases, emprega-se a LSTM empilhada..."*
- **O que quer dizer na prática:** $\odot$ é multiplicar dois vetores posição por posição, e a LSTM empilhada serve para modelar o comportamento dos gases ao longo do tempo.
- **Sugestão mais simples e natural:**
  - Linha 255: *"...$\odot$ representa a multiplicação elemento a elemento entre vetores, $f_t$ é a porta de esquecimento que controla o descarte de informações da célula anterior..."*
  - Linha 265: *"Para modelar o comportamento de séries temporais de gases com maior capacidade de representação, emprega-se a LSTM empilhada..."*

### Item P3.5 — "Desacoplar interações conjuntas" e "efeito principal univariado $\phi_{i,i}$" no TreeSHAP
- **Local:** [`fundamentacao.tex` (linhas 321–327)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L321-L327)
- **Como está hoje:**
  > *"Além da contribuição global, o TreeSHAP possibilita desacoplar as interações conjuntas entre pares de variáveis. Por meio da matriz de interações (`pred_interactions`), o efeito principal univariado de cada variável $\phi_{i,i}(x)$ é obtido isolando sua contribuição direta e subtraindo os termos cruzados de interação com as demais variáveis do modelo [...] identifica-se com precisão o ponto de inflexão em que a contribuição transita de negativa para positiva ($\phi_{i,i} = 0$). Esse cruzamento define o limiar marginal aprendido pelo algoritmo..."*
- **O que quer dizer na prática:** O valor SHAP normal de um gás mistura o efeito daquele gás sozinho com o efeito dele combinado com outros gases. A matriz de interações do TreeSHAP separa o efeito puro daquele gás sozinho ($\phi_{i,i}$). Quando olhamos onde esse efeito puro passa de negativo para positivo ($\phi_{i,i} = 0$), descobrimos a partir de qual valor aquele gás passa a indicar aquela falha para o XGBoost.
- **Sugestão mais simples e natural (sem dois pontos):**
  > *"Além da contribuição total de cada atributo, o TreeSHAP permite separar o efeito isolado de uma variável das suas combinações com as demais. Por meio da matriz de interações (`pred_interactions`), obtém-se o efeito individual $\phi_{i,i}(x)$ de cada gás ou razão após descontar as interações cruzadas com os outros atributos, conforme expresso na Equação (2.16). Ao relacionar esse efeito isolado $\phi_{i,i}(x)$ com o valor físico real da variável em ppm ou porcentagem, identifica-se o ponto exato em que sua contribuição passa de contrária ($\phi_{i,i} < 0$) para favorável ($\phi_{i,i} > 0$) àquela classe de falha ($\phi_{i,i} = 0$). Esse ponto de transição revela o limite prático aprendido pelo XGBoost, permitindo compará-lo diretamente com as fronteiras da norma IEC 60599 e do Triângulo de Duval."*

### Item P3.6 — "Modelagem por graus de pertinência difusos" e "compilação multicêntrica"
- **Local:** [`metodologia.tex` (linha 44)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/metodologia.tex#L44)
- **Como está hoje:**
  > *"Como refletem uma compilação multicêntrica de registros práticos, normas técnicas e ensaios laboratoriais, os dados exibem sobreposições físico-químicas naturais e casos limítrofes entre categorias --- por exemplo, na classe de falhas térmicas de alta temperatura (T3), a mediana observada para a razão $\text{C}_2\text{H}_4/\text{C}_2\text{H}_6$ é de apenas 0,25, distante do patamar teórico de 4,0 prescrito pela IEC 60599. Essa dispersão inerente a dados reais motivou Li, Wang e Song (2019) a proporem a modelagem por graus de pertinência difusos (membership degree), corroborando a estabilização empírica das métricas..."*
- **O que quer dizer na prática:** Como a base chinesa juntou dados de várias fontes diferentes (campo e laboratório), há muitos transformadores na fronteira entre duas falhas. No artigo original dessa base (Li, Wang e Song, 2019), os autores usaram lógica *fuzzy* justamente porque um transformador pode ter características mistas de duas classes.
- **Sugestão mais simples e natural (sem travessão nem dois pontos):**
  > *"Como essa base reúne registros de diferentes origens (inspeções em campo, literatura e laboratório), os dados apresentam sobreposições naturais e casos de fronteira entre tipos de falha. Na classe de falha térmica de alta temperatura (T3), por exemplo, a mediana da razão $\text{C}_2\text{H}_4/\text{C}_2\text{H}_6$ é de apenas 0,25, distante do valor superior a 4,0 exigido pela norma IEC 60599. Essa mistura entre classes em dados reais levou Li, Wang e Song (2019) a utilizarem lógica difusa (\textit{membership degree}) no estudo original da base, o que também explica por que o desempenho dos classificadores neste trabalho se estabiliza na faixa de 76\% a 78\% de $F_1$-macro."*

### Item P3.7 — "Extrapolação recursiva iterada" e "cinética das séries" na LSTM
- **Local:** [`metodologia.tex` (linhas 101 e 107)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/metodologia.tex#L101-L107)
- **Como está hoje:**
  > *"...sendo as previsões para horizontes estendidos ($h > 1$) obtidas por extrapolação recursiva iterada, na qual a predição $\hat{y}$ é sucessivamente realimentada na janela de entrada."* (L101)
  > *"...assegurando que as concentrações futuras reproduzam fielmente a cinética das séries antes do acoplamento ao classificador."* (L107)
- **Sugestão mais simples e natural (sem dois pontos):**
  > *"...sendo as previsões para múltiplos passos à frente ($h > 1$) obtidas de forma recursiva, em que cada valor estimado $\hat{y}$ é inserido no final da janela de 50 leituras (descartando-se a leitura mais antiga) para prever o passo seguinte."* (L101)
  > *"...assegurando que as estimativas futuras acompanhem com precisão a evolução das séries antes de alimentarem o classificador."* (L107)

### Item P3.8 — "Mutação perturbativa gaussiana" e "torneio determinístico" no IGA
- **Local:** [`metodologia.tex` (linha 123)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/metodologia.tex#L123)
- **Como está hoje:**
  > *"O cromossomo bipartido opera com esquemas específicos para cada segmento: para a máscara binária de seleção de atributos, utilizam-se cruzamento uniforme e mutação por inversão de bits com probabilidade $P_m = 0,10$; para o vetor de hiperparâmetros contínuos, aplicam-se o cruzamento BLX-$\alpha$ ($\alpha = 0,50$) e mutação perturbativa gaussiana com desvio padrão $\sigma = 0,10$. A seleção dos pais ocorre por torneio determinístico de tamanho $k = 3$, e as duas melhores soluções de cada geração são preservadas diretamente por elitismo estrito."*
- **Sugestão mais simples e natural (sem dois pontos):**
  > *"O cromossomo bipartido utiliza operadores específicos para cada parte. Na máscara binária de seleção de atributos, aplicam-se o cruzamento uniforme e a mutação por inversão de bits com probabilidade $P_m = 0,10$. No vetor de hiperparâmetros contínuos, aplicam-se o cruzamento BLX-$\alpha$ ($\alpha = 0,50$) e a mutação gaussiana (adição de ruído aleatório com média zero e desvio padrão $\sigma = 0,10$). A escolha dos pais é feita por torneio de tamanho $k = 3$, no qual três indivíduos são sorteados e seleciona-se o de maior aptidão, enquanto as duas melhores soluções de cada geração são mantidas diretamente para a próxima por elitismo."*

### Item P3.9 — "Parcimônia diagnóstica" e "complexidade operacional"
- **Local:** [`resultados.tex` (linha 121)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L121)
- **Como está hoje:**
  > *"Cumpre salientar, contudo, que o modelo de referência baseado unicamente em atributos observados (C5a) já atinge $F_1 = 0,8855$ com 9 variáveis, confirmando que a verdadeira parcimônia diagnóstica consiste em dispensar a complexidade operacional da rede neural recorrente e descartar atributos colineares."*
- **Sugestão mais simples e natural:**
  > *"Ressalta-se, contudo, que o modelo de referência treinado apenas com os atributos medidos no instante atual (C5a) já atinge $F_1 = 0,8855$ com 9 variáveis, mostrando que, para o diagnóstico imediato, é mais simples e prático utilizar diretamente as leituras atuais sem depender das projeções da rede LSTM."*

### Item P3.10 — "A nulidade está contida no intervalo de confiança"
- **Local:** [`resultados.tex` (linha 168)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L168)
- **Como está hoje:**
  > *"Em ambos os casos, a nulidade está contida no intervalo de confiança, demonstrando ausência de diferença estatisticamente detectável sob o nível de significância adotado entre os modelos baseados em gradiente de árvores."*
- **Sugestão mais simples e natural:**
  > *"Em ambos os casos, o valor zero está incluído dentro do intervalo de confiança de 95\% ($p > 0,05$), o que indica empate técnico e ausência de diferença estatisticamente significativa entre os modelos baseados em árvores."*

### Item P3.11 — "Limiar marginal extraído" e "contribuição marginal positiva" no TreeSHAP
- **Local:** [`resultados.tex` (linha 292)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L292)
- **Como está hoje:**
  > *"Por outro lado, os limiares marginais aprendidos para a porcentagem de acetileno [...] Essa moderação decorre da calibração empírica do classificador sobre dados reais de campo [...] Vale salientar que o ponto $\phi_{i,i} = 0$ indica o momento em que a variável passa a exercer contribuição marginal positiva, não se tratando de uma barreira geométrica rígida e estanque."*
- **Sugestão mais simples e natural:**
  > *"Por outro lado, os pontos de transição aprendidos pelo modelo para a porcentagem de acetileno em D1 e D2 ($\approx 8,0\%$ e $8,6\%$) e para a razão $\text{C}_2\text{H}_4/\text{C}_2\text{H}_6$ em T3 ($\approx 2,97$) são mais baixos que os limites fixos das normas (13\% e 4,0, respectivamente). Isso acontece porque o classificador se ajustou aos dados reais da base, na qual existem transformadores com laudo de descarga elétrica mesmo com teores menores de $\text{C}_2\text{H}_2$, ou falhas térmicas T3 com menor proporção de etileno. Nesses casos de T3, o modelo compensa utilizando também o percentual de etano (\texttt{ratio\_C2H6\_TH1}), como visto na Figura~\ref{fig:shap_heatmap}. Cabe lembrar que o ponto $\phi_{i,i} = 0$ marca apenas o valor a partir do qual aquele gás passa a somar pontos a favor da falha nas árvores, e não uma linha divisória única como nas normas."*

### Item P3.12 — "Sobrecarga arquitetural e operacional"
- **Local:** [`conclusao.tex` (linha 23)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/conclusao.tex#L23)
- **Como está hoje:**
  > *"O resultado negativo documentado mostra que, para a base avaliada, o acoplamento da rede recorrente não traz ganho que justifique a sobrecarga arquitetural e operacional de treinar e manter modelos neurais adicionais para o diagnóstico pontual."*
- **Sugestão mais simples e natural:**
  > *"Esse resultado mostra na prática que, para a base avaliada, incluir as previsões da rede LSTM como entrada do XGBoost não traz ganho que justifique o custo extra de treinar e manter os dois modelos acoplados apenas para classificar o estado atual do transformador."*

---

## Parte B — Remoção de "Dois Pontos Explicativos" (`:`) no Meio das Frases

Abaixo estão todos os trechos do texto atual onde dois pontos (`:`) foram usados no meio de uma frase para explicar algo (construção típica de IA), com a substituição direta por conectivos naturais:

| Item | Localização | Como está hoje (com `:`) | Reescrita natural (sem `:`) |
| :--- | :--- | :--- | :--- |
| **P3.13** | [`introducao.tex` (L18)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/introducao.tex#L18) | *"enfrenta, contudo, um desafio comum na área: a heterogeneidade das bases de dados abertas disponíveis."* | *"enfrenta, contudo, a heterogeneidade das bases de dados abertas disponíveis na área."* |
| **P3.14** | [`fundamentacao.tex` (L9 e L30)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L9-L30) | L9: *"três características operacionais são particularmente relevantes: a composição do sistema..."* <br> L30: *"apresenta grande desbalanceamento: ocorrências de arcos elétricos severos..."* | L9: *"destacam-se três características operacionais principais, que são a composição do sistema..."* <br> L30: *"apresenta grande desbalanceamento, pois ocorrências de arcos elétricos severos..."* |
| **P3.15** | [`fundamentacao.tex` (L289)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L289) | *"estruturado por um cromossomo bipartido: uma máscara binária (que define quais variáveis são utilizadas) e um vetor..."* | *"estruturado por um cromossomo bipartido composto por uma máscara binária (que define quais variáveis são utilizadas) e por um vetor..."* |
| **P3.16** | [`metodologia.tex` (L64, L66 e L93)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/metodologia.tex#L64-L93) | L64: *"deslocamento unitário: $x_{\text{log}} = \ln(x + 1)$ [...] normalização Min-Max: $(x_{\text{log}} - x_{\text{min}})..."* <br> L66: *"É fundamental destacar a ordem rigorosa de processamento: todas as razões..."* <br> L93: *"frequência relativa no treino: $w_c = N / (C \cdot N_c)$..."* | L64: *"deslocamento unitário $x_{\text{log}} = \ln(x + 1)$ [...] normalização Min-Max dada por $(x_{\text{log}} - x_{\text{min}})..."* <br> L66: *"Ressalta-se que todas as razões físico-químicas normativas..."* <br> L93: *"frequência relativa no treino, segundo a relação $w_c = N / (C \cdot N_c)$..."* |
| **P3.17** | [`resultados.tex` (L119)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L119) | *"elucida igualmente o comportamento da linha ``C1 sem LSTM (pós-hoc)'' na Tabela 5: a simples exclusão forçada..."* | *"explica também o comportamento da linha ``C1 sem LSTM (pós-hoc)'' na Tabela 5, visto que a simples exclusão forçada..."* |
| **P3.18** | [`resultados.tex` (L131)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L131) | *"boa capacidade de acerto nas classes avaliadas: 709 acertos [...] Esse padrão é condizente com a física do problema: sobreaquecimentos térmicos leves..."* | *"boa capacidade de acerto nas classes avaliadas, registrando 709 acertos [...] Esse padrão é condizente com a física do problema, pois sobreaquecimentos térmicos leves..."* |
| **P3.19** | [`resultados.tex` (L170 e L182)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L170-L182) | L170: *"ausência de auditoria de duplicatas: em bases como a do Track B..."* <br> L182: *"Observam-se 10 erros entre descargas de baixa energia (D1) e alta energia (D2): 6 amostras..."* | L170: *"ausência de auditoria de duplicatas, uma vez que, em bases como a do Track B..."* <br> L182: *"Observam-se 10 erros entre descargas de baixa energia (D1) e alta energia (D2), sendo 6 amostras..."* |
| **P3.20** | [`resultados.tex` (L215)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L215) | *"A primeira causa decorre do próprio escopo das normas: tanto o Triângulo de Duval quanto as razões da IEC..."* | *"A primeira causa decorre do próprio escopo das normas, visto que tanto o Triângulo de Duval quanto as razões da IEC..."* |
| **P3.21** | [`resultados.tex` (L306)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L306) | *"concentram-se em duas regiões críticas: nas imediações das fronteiras..."* | *"concentram-se principalmente nas imediações das fronteiras..."* |
| **P3.22** | [`resultados.tex` (L334 e L336)](file:///C:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L334-L336) | L334: *"oscila entre as duas variáveis temporais do mesmo gás: na semente 42..."* <br> L336: *"sem fornecer informação discriminativa complementar: o modelo aloca o crédito..."* | L334: *"oscila entre as duas variáveis temporais do mesmo gás. Na semente 42..."* <br> L336: *"sem fornecer informação discriminativa complementar, fazendo com que o modelo aloque o crédito..."* |
