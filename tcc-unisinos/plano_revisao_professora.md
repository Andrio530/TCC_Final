# Plano de Implementação — Correções Solicitadas pela Orientadora

> **Data da Auditoria:** 30 de outubro de 2026  
> **Escopo:** Verificação ponto a ponto dos apontamentos da professora sobre o texto e as figuras do TCC (`tcc-unisinos/`), identificando o que ainda está em desacordo na versão atual e definindo o plano exato de implementação.

---

## 1. Matriz de Diagnóstico dos Apontamentos da Professora

| # | Apontamento da Professora | Local no LaTeX Atual | Status Atual | Diagnóstico Técnico |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **P.14 (Objetivos Específicos):**<br>*"Analisar a coerência físico-química dos padrões de decisão do XGBoost em relação às normas IEC 60599 e ao Triângulo de Duval, empregando valores de Shapley por meio do algoritmo TreeSHAP."*<br>• Citar referências para a norma e para os algoritmos;<br>• Mencionar o nome da norma por extenso. | `introducao.tex`<br>(L34–44, Seção 1.3) | **EM DESACORDO** | A linha 43 de `introducao.tex` está exatamente com essa redação sem as citações (`\cite{...}`) e sem o título por extenso da norma IEC 60599. Os demais itens dos objetivos (L40–42) também mencionam LSTM, XGBoost e IGA sem citação. |
| **2** | **P.16 (Fundamentação Teórica):**<br>• Citar referência para a Lei de Faraday. | `fundamentacao.tex`<br>(L9, Seção 2.1) | **EM DESACORDO** | Na linha 9 de `fundamentacao.tex`, a frase sobre a Lei de Faraday está acompanhada de `\cite{tightiz2020}` (um artigo sobre ANFIS que não é referência de eletromagnetismo/máquinas elétricas, ou estava sem referência na versão lida pela professora). Deve-se citar literatura canônica de máquinas elétricas/física (ex.: Chapman, 2013 / Faraday, 1832). |
| **3** | **P.28 (Figura 6 — IGA):**<br>• Ampliar a Figura 6 (*Fluxo operacional do Algoritmo Genético Melhorado para busca conjunta*). As letras ficaram muito pequenas em 100%. | `fundamentacao.tex`<br>(L277–283, `fig:IGA`) | **EM DESACORDO** | A figura está dimensionada com `width=0.55\textwidth` (`fundamentacao.tex` L279). Por ser um fluxograma vertical, 55% da largura do texto reduz excessivamente as fontes internas das caixas. |
| **4** | **P.34 (Figura 7 — Arquitetura):**<br>• Ampliar a Figura 7 (*Fluxograma metodológico do trabalho estruturado na divisão entre os eixos Track A e Track B*). As letras ficaram muito pequenas em 100%. | `metodologia.tex`<br>(L11–17, `fig:arquitetura`) | **EM DESACORDO** | A figura está travada em `height=9.5cm, keepaspectratio` (`metodologia.tex` L13). Como o diagrama tem proporção vertical `798 × 1448`, limitar a altura a 9,5 cm reduz a largura a apenas ~5,2 cm (~32% da largura útil da página), deixando as fontes minúsculas em 100%. |
| **5** | **P.49 (Figura 9 — Trade-off):**<br>• Ampliar a Figura 9 (*Relação entre complexidade dimensional e desempenho diagnóstico nos Tracks A e B*). As letras ficaram muito pequenas em 100%. | `resultados.tex` (L123–129) e `generate_rich_plots.py` (L27–51) | **EM DESACORDO** | No script `generate_rich_plots.py` (L46), a legenda foi posicionada **fora** do gráfico à direita (`bbox_to_anchor=(1.02, 1)`), fazendo com que ~40% da largura da imagem seja espaço em branco sob a legenda externa e encolhendo a área útil do gráfico e das fontes (`fontsize=9`) quando inserida no LaTeX. |
| **6** | **Em todo o trabalho (Coesão entre Capítulos e Seções):**<br>• Conectar os capítulos com um breve fechamento de um parágrafo, mencionando o assunto a ser tratado no próximo capítulo;<br>• Fazer o mesmo quando o assunto mudar muito de uma seção para outra. | `introducao.tex`, `fundamentacao.tex`, `metodologia.tex`, `resultados.tex` | **EM DESACORDO** | Os Capítulos 2, 3 e 4 terminam abruptamente sem parágrafo de transição para o capítulo seguinte (o Cap. 3 termina dentro de um `\end{enumerate}`!), e diversas seções mudam de assunto (ex.: após figuras/listas ou entre blocos temáticos) sem frase/parágrafo de conexão. |

---

## 2. Plano Detalhado de Implementação

### Item 1 — Referências e Nome por Extenso da Norma nos Objetivos (`introducao.tex`, L40–44)
- **Arquivo:** `tcc-unisinos/introducao.tex` (Seção 1.3 — Objetivos, linhas 40–44).
- **Ação:**
  1. No quarto item dos objetivos específicos (linha 43), expandir o nome da norma IEC 60599 por extenso (*IEC 60599 — Mineral oil-filled electrical equipment in service – Guidance on the interpretation of dissolved and free gases analysis*, ou com sua tradução técnica) e inserir as referências formais para a norma (`\cite{iec60599_2022}`), para o Triângulo de Duval (`\cite{duval2002_review}`), para o XGBoost (`\cite{chen2016_xgboost}`), para os valores de Shapley (`\cite{shapley1953,lundberg2017}`) e para o algoritmo TreeSHAP (`\cite{lundberg2020}`).
  2. Nos três itens anteriores dos objetivos específicos (linhas 40–42), incluir também as referências de origem dos algoritmos quando citados: LSTM (`\cite{hochreiter1997long}`), XGBoost (`\cite{chen2016_xgboost}`) e IGA (`\cite{wu2021}`).
- **Proposta de Redação (L38–44 de `introducao.tex`):**
  ```latex
  \begin{itemize}
      \item Auditar e preparar bases de dados industriais de DGA, tratando amostras duplicadas e organizando os experimentos em dois eixos analíticos (Track A e Track B) de acordo com a disponibilidade de variáveis temporais e físico-químicas;
      \item Implementar e avaliar um módulo de previsão temporal baseado em redes neurais recorrentes \textit{Long Short-Term Memory} (LSTM) \cite{hochreiter1997long} para estimar a evolução de concentrações de gases em diferentes horizontes futuros;
      \item Avaliar experimentalmente, por meio de testes pareados com reamostragem estatística (\textit{bootstrap}), se a inclusão das projeções da LSTM melhora o desempenho do classificador \textit{Extreme Gradient Boosting} (XGBoost) \cite{chen2016_xgboost} no diagnóstico pontual;
      \item Avaliar a capacidade do Algoritmo Genético Melhorado (IGA) \cite{wu2021} em selecionar subconjuntos reduzidos de variáveis para o XGBoost \cite{chen2016_xgboost}, buscando simplificar o modelo sem perda de desempenho diagnóstico;
      \item Analisar a coerência físico-química dos padrões de decisão do XGBoost \cite{chen2016_xgboost} em relação à norma técnica IEC 60599 (\textit{Mineral oil-filled electrical equipment in service -- Guidance on the interpretation of dissolved and free gases analysis}) \cite{iec60599_2022} e ao método do Triângulo de Duval \cite{duval2002_review}, empregando valores de Shapley \cite{shapley1953,lundberg2017} por meio do algoritmo TreeSHAP \cite{lundberg2020}.
  \end{itemize}
  ```

---

### Item 2 — Referência para a Lei de Faraday (`fundamentacao.tex` L9 e `referencias.bib`)
- **Arquivos:** `tcc-unisinos/referencias.bib` e `tcc-unisinos/fundamentacao.tex` (linha 9).
- **Ação:**
  1. Adicionar em `referencias.bib` a referência clássica de máquinas elétricas/transformadores de Stephen J. Chapman (*Fundamentos de Máquinas Elétricas*, 5. ed., McGraw-Hill / AMGH, 2013 — chave `chapman2013`).
  2. Em `fundamentacao.tex` (L9), substituir `\cite{tightiz2020}` por `\cite{chapman2013}` na explicação da Lei de Faraday da indução eletromagnética.
- **Proposta de Redação (`fundamentacao.tex` L9):**
  ```latex
  O funcionamento de um transformador baseia-se na Lei de Indução Eletromagnética de Faraday, segundo a qual uma variação temporal de fluxo magnético através de um circuito fechado induz uma força eletromotriz nos enrolamentos acoplados magneticamente ao núcleo \cite{chapman2013}.
  ```

---

### Item 3 — Ampliação da Figura 6 (`fig:IGA` em `fundamentacao.tex` L277–283)
- **Arquivo:** `tcc-unisinos/fundamentacao.tex` (linhas 277–283) e `tcc-unisinos/figuras/IGA.png`.
- **Diagnóstico:** A imagem `IGA.png` tem resolução alta (`1081 × 1359 px`), mas como estava com `width=0.55\textwidth` e possui margens laterais e superiores, ficou pequena na página.
- **Ação:**
  1. Recortar automaticamente as bordas excedentes de `IGA.png` via Python (preservando um pequeno respiro limpo) para maximizar a área útil das caixas e textos.
  2. Ampliar no LaTeX (`fundamentacao.tex` L279) de `width=0.55\textwidth` para `width=0.88\textwidth` (ou `height=0.78\textheight, keepaspectratio`), aumentando o tamanho visual das letras em cerca de **60%**.

---

### Item 4 — Ampliação da Figura 7 (`fig:arquitetura` em `metodologia.tex` L11–17)
- **Arquivo:** `tcc-unisinos/metodologia.tex` (linhas 11–17) e `tcc-unisinos/figuras/Arquitetura.pdf`.
- **Diagnóstico:** `Arquitetura.pdf` estava com `height=9.5cm, keepaspectratio`, o que comprimia todo o fluxograma vertical de 4 estágios em menos da metade da página (~5,2 cm de largura).
- **Ação:**
  1. Alterar em `metodologia.tex` (L13) de `\includegraphics[height=9.5cm, keepaspectratio]{figuras/Arquitetura.pdf}` para `\includegraphics[height=0.84\textheight, keepaspectratio]{figuras/Arquitetura.pdf}` (ou `width=0.95\textwidth, height=0.84\textheight, keepaspectratio`), dedicando a página à visualização completa do fluxograma e **dobrando (+100%)** o tamanho de todas as fontes em visualização 100%.

---

### Item 5 — Ampliação e Reformatação da Figura 9 (`fig:tradeoff_complexidade` em `resultados.tex` L123–129)
- **Arquivos:** `tcc-transformer-fault/src/iga_xgboost/generate_rich_plots.py` (L19–51), `tcc-unisinos/figuras/xgb_tradeoff_complexity.png` e `tcc-unisinos/resultados.tex` (L125).
- **Diagnóstico:** A legenda estava posicionada fora da área do gráfico à direita (`bbox_to_anchor=(1.02, 1)`), desperdiçando ~40% da largura horizontal da imagem e encolhendo o gráfico e as fontes (`fontsize=9`).
- **Ação:**
  1. Atualizar `generate_rich_plots.py` para:
     - Posicionar a legenda **dentro** da área livre do gráfico (ex.: `loc='center left'` na faixa central vazia entre $F_1 = 0,81$ e $0,87$, ou no topo interno), eliminando a coluna lateral vazia;
     - Aumentar o tamanho das fontes: título para `14pt`, eixos X e Y para `13pt`, *ticks* dos eixos para `11.5pt`, caixas de anotação dos pontos para `11.5pt` e legenda para `11pt`;
     - Regerar `xgb_tradeoff_complexity.png` em 300 DPI diretamente em `tcc-unisinos/figuras/`.
  2. Em `resultados.tex` (L125), ampliar de `width=0.85\textwidth` para `width=0.95\textwidth`.

---

### Item 6 — Parágrafos de Fechamento e Conexão entre Capítulos e Seções

#### 6.1. Conexões no Final de Cada Capítulo (Fechamento de Capítulo $\rightarrow$ Próximo Capítulo)

1. **Final do Capítulo 1 (`introducao.tex`, após L54):**
   - **Situação atual:** Termina listando o conteúdo do Capítulo 5 na Seção 1.4.
   - **Adição proposta:**
     > *"Definidos o contexto operacional, a justificativa e os objetivos da investigação, o próximo capítulo apresenta a fundamentação teórica que embasa os fenômenos de degradação isolante, os critérios normativos de DGA e os algoritmos computacionais empregados neste estudo."*

2. **Final do Capítulo 2 (`fundamentacao.tex`, após L363):**
   - **Situação atual:** Termina resumindo o posicionamento do trabalho frente à literatura, sem mencionar o Capítulo 3.
   - **Adição proposta:**
     > *"No próximo capítulo, detalham-se os procedimentos metodológicos estruturados para operacionalizar essa investigação, abrangendo a arquitetura experimental em dois eixos (Track A e Track B), a auditoria e o pré-processamento das bases de dados, a engenharia de atributos, a modelagem preditiva e os protocolos de validação estatística e de interpretabilidade."*

3. **Final do Capítulo 3 (`metodologia.tex`, após L168):**
   - **Situação atual:** Termina de forma abrupta no `\end{enumerate}` da Seção 3.7 (Protocolo de Interpretabilidade).
   - **Adição proposta:**
     > *"Estabelecidos os conjuntos de dados auditados, a arquitetura dos modelos preditivos e os critérios de validação estatística e físico-química, o Capítulo~\ref{cap:resultados} apresenta e discute os resultados experimentais obtidos nos eixos Track A e Track B."*

4. **Final do Capítulo 4 (`resultados.tex`, após L330):**
   - **Situação atual:** Termina na análise numérica da migração de crédito SHAP na Seção 4.5.6.
   - **Adição proposta:**
     > *"Em conjunto, os resultados experimentais dos eixos Track A e Track B demonstram a viabilidade da simplificação dimensional via IGA e a consistência físico-química capturada pelo TreeSHAP, ao mesmo tempo em que evidenciam as limitações práticas da inclusão de projeções temporais em séries de alta inércia e o teto de desempenho imposto pela sobreposição entre classes em dados estáticos. Com base nessas evidências, o Capítulo~\ref{cap:conclusao} sintetiza as conclusões gerais do trabalho, sistematiza as limitações identificadas e propõe direcionamentos para pesquisas futuras."*

---

#### 6.2. Conexões nas Transições entre Seções (Mudanças de Assunto intra-capítulo)

1. **No Capítulo 2 (`fundamentacao.tex`):**
   - **Seção 2.1 $\rightarrow$ Seção 2.2 (após L31, fim de *Transformadores de Potência*):**
     - *Problema:* Termina dentro de um `\end{itemize}` sobre falhas combinadas e passa direto para `\section{Métodos Normativos de Diagnóstico em DGA}`.
     - *Conexão proposta:*
       > *"Compreendidos os mecanismos físico-químicos de geração de gases e a taxonomia das falhas internas, a seção seguinte apresenta os métodos normativos tradicionalmente adotados pela engenharia de manutenção para interpretar essas concentrações gasosas."*
   - **Seção 2.2 $\rightarrow$ Seção 2.3 (após L91, fim de *Métodos Normativos*):**
     - *Problema:* Termina apontando as limitações da IEC 60599.
     - *Conexão proposta:*
       > *"Para superar essas restrições geométricas e a ocorrência de diagnósticos inconclusivos, a seção a seguir descreve os algoritmos de aprendizado de máquina e de otimização evolutiva aplicados à classificação automatizada e à previsão temporal em DGA."*
   - **Seção 2.3 $\rightarrow$ Seção 2.4 (após L285, fim de *Algoritmos de Aprendizado de Máquina / IGA*):**
     - *Problema:* Termina explicando o elitismo do IGA e muda abruptamente para `\section{Interpretabilidade em Aprendizado de Máquina (XAI)}`.
     - *Conexão proposta:*
       > *"Embora a combinação entre ensembles de árvores e otimização evolutiva proporcione elevado poder discriminativo, sua adoção segura em diagnósticos industriais requer transparência sobre os critérios internos de decisão, tema abordado na próxima seção por meio das técnicas de interpretabilidade em aprendizado de máquina."*
   - **Seção 2.4 $\rightarrow$ Seção 2.5 (após L321, fim de *Interpretabilidade / TreeSHAP*):**
     - *Problema:* Termina explicando o ponto de inflexão $\phi_{i,i} = 0$ e entra direto em `\section{Trabalhos Relacionados}`.
     - *Conexão proposta:*
       > *"Apresentados os fundamentos teóricos dos modelos preditivos, da busca evolutiva e da decomposição por valores de Shapley, a seção seguinte analisa como os trabalhos recentes da literatura têm integrado essas técnicas no diagnóstico de transformadores de potência."*

2. **No Capítulo 3 (`metodologia.tex`):**
   - **Seção 3.1 $\rightarrow$ Seção 3.2 (após L17, fim de *Desenho Metodológico*):**
     - *Problema:* Termina na Figura 7 sem texto pós-figura antes de `\section{Bases de Dados e Procedimentos de Auditoria}`.
     - *Conexão proposta:*
       > *"Para viabilizar a execução dos dois eixos ilustrados na Figura~\ref{fig:arquitetura}, a próxima seção detalha a proveniência, a distribuição de classes e os procedimentos de auditoria e deduplicação aplicados às bases de dados de cada trilha experimental."*
   - **Seção 3.2 $\rightarrow$ Seção 3.3 (após L50, fim de *Bases de Dados*):**
     - *Problema:* Termina discutindo o descarte de bases de Monte Carlo e muda para `\section{Pré-processamento e Engenharia de Atributos}`.
     - *Conexão proposta:*
       > *"Definidas e saneadas as bases de dados dos eixos Track A e Track B, a seção seguinte descreve os procedimentos de normalização de escala, engenharia de atributos físico-químicos e balanceamento de classes aplicados previamente ao treinamento dos modelos."*
   - **Seção 3.3 $\rightarrow$ Seção 3.4 (após L91, fim de *Pré-processamento / CustomSMOTE*):**
     - *Problema:* Termina explicando o `CustomSMOTE` na base estática do Track B e muda repentinamente para `\section{Módulo de Previsão Temporal}` (LSTM do Track A).
     - *Conexão proposta:*
       > *"Concluída a estruturação dos espaços de atributos e o tratamento do desbalanceamento, as próximas seções especificam a configuração dos algoritmos de modelagem, iniciando pela arquitetura e pelo protocolo de treinamento da rede neural recorrente LSTM responsável pelas projeções temporais do Track A."*
   - **Seção 3.4 $\rightarrow$ Seção 3.5 (após L101, fim de *Módulo de Previsão Temporal*):**
     - *Problema:* Termina na escolha de $h=14$ e $h=28$ e muda para `\section{Algoritmo Genético Melhorado e Otimização do XGBoost}`.
     - *Conexão proposta:*
       > *"Uma vez geradas as projeções temporais pela rede LSTM e consolidados os vetores de atributos candidatos, a seção a seguir detalha o Algoritmo Genético Melhorado (IGA) empregado para selecionar as variáveis mais relevantes e calibrar os hiperparâmetros do classificador XGBoost."*
   - **Seção 3.5 $\rightarrow$ Seção 3.6 (após L119, fim de *IGA e Otimização do XGBoost*):**
     - *Problema:* Termina nos critérios de parada do IGA e muda para `\section{Protocolo Experimental e Rigor Estatístico}`.
     - *Conexão proposta:*
       > *"Para avaliar de forma rigorosa o desempenho das configurações obtidas pelo IGA e testar a hipótese de hibridização temporal sem viés amostral, a próxima seção define o desenho dos experimentos de ablação e os testes estatísticos de significância."*
   - **Seção 3.6 $\rightarrow$ Seção 3.7 (após L155, fim de *Protocolo Experimental*):**
     - *Problema:* Termina na explicação da validação cruzada $5 \times 5$ e muda para `\section{Protocolo de Interpretabilidade e Validação Físico-Química}`.
     - *Conexão proposta:*
       > *"Complementando a avaliação quantitativa de desempenho estatístico, a seção seguinte estabelece o protocolo de interpretabilidade via TreeSHAP destinado a auditar a coerência físico-química das regras aprendidas pelo classificador frente às normas técnicas."*

3. **No Capítulo 4 (`resultados.tex`):**
   - **Seção 4.2 $\rightarrow$ Seção 4.3 (após L139, fim de *Resultados do Track A*):**
     - *Problema:* Termina na Figura 10 (Matriz de confusão do Track A) e salta direto para `\section{Resultados do Track B: Diagnóstico Multiclasse e Limites dos Dados}`.
     - *Conexão proposta:*
       > *"Encerrada a avaliação da hipótese temporal e da redução dimensional sobre os quatro estados operacionais do Track A, a seção seguinte investiga o desempenho diagnóstico do modelo IGA-XGBoost no cenário estático multiclasse de sete categorias da norma IEC 60599 (Track B), confrontando-o com métodos normativos clássicos e configurações da literatura."*
   - **Seção 4.3 $\rightarrow$ Seção 4.4 (após L180, fim de *Resultados do Track B*):**
     - *Problema:* Termina comentando os acertos nas classes Normal e T3 da matriz de confusão e muda para `\section{Auditoria Normativa: Divergência dos Métodos Tradicionais}`.
     - *Conexão proposta:*
       > *"Além da estabilidade observada entre os modelos baseados em árvores, os resultados da Tabela~\ref{tab:track_b_desempenho} evidenciaram um baixo desempenho absoluto dos métodos normativos tradicionais sobre a base avaliada, cuja investigação detalhada é apresentada na seção a seguir."*
   - **Seção 4.4 $\rightarrow$ Seção 4.5 (após L209, fim de *Auditoria Normativa*):**
     - *Problema:* Termina discutindo o ajuste supervisionado aos rótulos de T3 e muda para `\section{Interpretabilidade Físico-Química via SHAP}`.
     - *Conexão proposta:*
       > *"Diante dessa diferença de comportamento entre as regras normativas estritas e o ajuste supervisionado, torna-se fundamental inspecionar internamente os critérios aprendidos pelo XGBoost, análise conduzida na próxima seção por meio da decomposição de valores de Shapley (TreeSHAP)."*

---

## 3. Ordem de Execução (Aguardando Aprovação)

1. **Passo 1 (`referencias.bib` + `introducao.tex` + `fundamentacao.tex`):** Inserir a referência da Lei de Faraday (`chapman2013`), atualizar os objetivos específicos na p. 14 com o nome por extenso da norma IEC 60599 e as citações de todos os métodos/algoritmos.
2. **Passo 2 (Figuras 6, 7 e 9):**
   - Recortar margens excedentes de `IGA.png` e ampliar `fig:IGA` em `fundamentacao.tex` (`width=0.88\textwidth`);
   - Ampliar `fig:arquitetura` em `metodologia.tex` (`height=0.84\textheight, keepaspectratio`);
   - Regerar `xgb_tradeoff_complexity.png` com legenda interna e fontes ampliadas em `generate_rich_plots.py` e ajustar `fig:tradeoff_complexidade` para `width=0.95\textwidth` em `resultados.tex`.
3. **Passo 3 (Parágrafos de Conexão entre Capítulos e Seções):** Inserir os fechamentos de capítulo e transições de seção listados no Item 6 em `introducao.tex`, `fundamentacao.tex`, `metodologia.tex` e `resultados.tex`.

