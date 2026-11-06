# Relatório de Auditoria de Referências Bibliográficas

## 1. Matriz de Conformidade

| Referência Citada | Páginas / Local | Status | Severidade | Resumo do Diagnóstico |
| :--- | :--- | :--- | :--- | :--- |
| **ZHU et al., 2023** | p. 23, 24, 32, 69 | **FANTASMA** | **CRÍTICA** | Referência inexistente no IEEE TPWRS. Trata-se de alucinação bibliográfica sobre o artigo do *Informer* (Zhou et al., 2021), cujo PDF real está na pasta. |
| **WU et al., 2021** | p. 14, 23, 24, 29, 32, 39, 43, 51, 52, 69 | **DIVERGENTE** | **CRÍTICA** | O texto afirma que Wu et al. obtiveram >99% "sobre essa mesma base" devido a vazamento de duplicatas (15,5%). Na realidade, Wu et al. usaram uma base distinta de apenas 150 amostras (Kirkbas et al., 2020). |
| **WANG, LITTLER e LIU, 2023** | p. 14, 32, 37, 69 | **DIVERGENTE** | **ALTA** | O texto afirma que os autores "reportaram $R^2 > 0,90$" e "não avaliaram o acoplamento a classificadores". Na realidade, o artigo não usa $R^2$ (usa MAE/RMSE/MSE) e propõe explicitamente modelos acoplados a 4 classificadores (SVM, KNN, CART, ANN). |
| **LI, WANG e SONG, 2019** | p. 36, 37, 69 | **DIVERGENTE** | **ALTA** | Atribui a Li et al. a afirmação sobre laudos de inspeção/desparafusamento e a justificativa do teto de 76–78% para modelos de ML. O artigo propõe graus de pertinência, mas o PDF não está na pasta para conferência textual dos laudos. |
| **IBRAHIM, GHONEIM e TAHA, 2018** | p. 36, 68 | **DIVERGENTE** | **MÉDIA** | O texto confunde a ferramenta DGALab com o repositório de dados, afirmando que a base chinesa de 2019 estava catalogada no DGALab de 2018 hospedado no repositório de Alan. |
| **ALAN, 2021** | p. 36, 68 | **DIVERGENTE** | **MÉDIA** | A entrada BibTeX inventou a inicial "C." para o autor ("Alan, C.") a partir do usuário do GitHub `alan-456`. Não é autoria acadêmica formal, mas sim um repositório de código/dados. |
| **TIGHTIZ et al., 2020** | p. 12, 17, 69 | **DIVERGENTE** | **BAIXA** | Citado na p. 17 para fundamentar a Lei de Faraday da indução eletromagnética (princípio físico fundamental de 1831). Artigo específico de ANFIS usado para física clássica. PDF ausente na pasta. |
| **AL-SAKINI et al., 2025** | p. 14, 18, 22, 24, 32, 33, 68 | **CONFORME** | **NENHUMA** | Métricas (>95% em Random Forest), classes e modelos comparados conferem rigorosamente com o artigo em `applsci-15-00118.pdf`. |
| **ALI et al., 2023** | p. 12, 17, 18, 20, 22, 23, 24, 28, 32, 33, 68 | **CONFORME** | **NENHUMA** | Dados de degradação térmica do óleo, geração de gases e diagrama esquemático conferem rigorosamente com `1-s2.0-S0378779622011130-main.pdf`. |
| **KATSER et al., 2022** | p. 23, 34, 35, 45, 48, 66, 68 | **CONFORME** | **NENHUMA** | Volume de 3.000 séries, divisão 2.100/900, 420 passos a cada 12h e 4 gases conferem rigorosamente com `2211.11013v1.pdf`. |
| **CHEN e GUESTRIN, 2016** | p. 25, 68 | **CONFORME** | **NENHUMA** | Formulação analítica da função objetivo, expansão de Taylor em 2ª ordem e regularização conferem com `1603.02754v3.pdf`. |
| **FRIEDMAN, 2001** | p. 25, 68 | **CONFORME** | **NENHUMA** | Fundamentação teórica do Gradient Boosting machine confere com `document.pdf`. |
| **HOCHREITER e SCHMIDHUBER, 1997** | p. 26, 27, 68 | **CONFORME** | **NENHUMA** | Estrutura de memória de longo prazo, fluxo de gradiente e equações da célula LSTM conferem com `2604.pdf`. |
| **LECUN et al., 1998** | p. 23, 24, 68 | **CONFORME** | **NENHUMA** | Citação metodológica clássica de redes neurais e aprendizado baseado em gradiente confere com `726791.pdf`. |
| **LUNDBERG e LEE, 2017** | p. 30, 69 | **CONFORME** | **NENHUMA** | Definição da estrutura SHAP e axiomas aditivos de explicação local conferem com o paper do NeurIPS 2017. |
| **LUNDBERG et al., 2020** | p. 44, 55, 69 | **CONFORME** | **NENHUMA** | Complexidade polinomial do TreeSHAP e aditividade no espaço de margens (*log-odds*) conferem com `1905.04610v1.pdf`. |
| **SHAPLEY, 1953** | p. 30, 69 | **CONFORME** | **NENHUMA** | Formulação axiomática clássica do valor de Shapley em teoria dos jogos confere com `P295.pdf`. |
| **DUVAL, 2002** | p. 19, 68 | **CONFORME** | **NENHUMA** | Formulação clássica das coordenadas do Triângulo 1 de Duval. Conteúdo técnico plenamente alinhado com as normas. PDF ausente na pasta. |
| **IEC, 2022 (IEC 60599)** | p. 3, 4, 10, 14, 15, 17, 19, 21, 31, 34, 37, 39, 43, 51, 52, 54, 55, 59, 64 | **CONFORME** | **NENHUMA** | Limites normativos de razões, zonas do Triângulo de Duval e codificação de falhas conferem com a norma técnica. PDF ausente na pasta. |

---

## 2. Diagnóstico Detalhado e Análise de Discrepâncias

### [DISC-01] - ZHU et al., 2023 (Referência Inexistente / Alucinação)
- **Localização no TCC:** Página 23 (Tabela de modelos, [fundamentacao.tex L129](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L129)), Página 24 (Tabela 1, [L166](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L166)), Página 32 (Tabela 4, [L346](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L346)) e Página 32 (Texto de correlatos, [L361](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L361)).
- **Trecho Citado no TCC:**
  > "Por fim, o uso de arquiteturas Transformer em séries temporais de transformadores, explorado por \citeonline{zhu2023}, confirma a utilidade dos mecanismos de autoatenção para sequências longas, mas esbarra na exigência de grandes volumes de dados de treinamento, raramente disponíveis em bases públicas abertas de DGA." (p. 32, Seção 2.8)
  > "Transformer & Zhu et al. (2023) & Mecanismos de atenção em horizontes longos & Exigência de grandes volumes de dados, indisponíveis em DGA aberta" (p. 24, Tabela 1)
- **Evidência no PDF Original:**
  > Não existe artigo de Zhu et al. com esse título no volume 38, n. 4 do *IEEE Transactions on Power Systems* (as páginas 3245–3255 pertencem a outro trabalho). No entanto, na pasta `artigos/`, encontra-se o arquivo `17325-13-20819-1-2-20210518.pdf`, que é o artigo original do **Informer**:
  > *"Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"*, por Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong e Wancai Zhang (*Proceedings of the AAAI Conference on Artificial Intelligence*, v. 35, n. 12, p. 11106–11115, 2021).
- **Problema Identificado:** A citação foi gerada por alucinação, copiando o título do paper do Informer (Zhou et al., 2021) e inventando autores e periódico fictícios.
- **Ação Corretiva Proposta:**
  - **Texto sugerido para substituição no LaTeX (Seção 2.8, p. 32):**
    > "Por fim, o uso de arquiteturas baseadas em mecanismos de autoatenção para séries temporais longas, a exemplo do modelo Informer proposto por \citeonline{zhou2021_informer}, destaca o potencial dos Transformers para capturar dependências temporais estendidas, porém essas redes demandam volumes massivos de dados para treinamento, inviabilizando sua aplicação direta sobre as bases públicas abertas de DGA atualmente disponíveis."
  - **Ajuste na Tabela 1 (p. 24) e Tabela 4 (p. 32):**
    Substituir `Zhu et al. (2023)` por `Zhou et al. (2021)` e o foco para `Informer / Transformers em Séries Temporais`.
  - **Ajuste no `referencias.bib`:**
    Substituir a entrada `@article{zhu2023, ...}` por:
    ```bibtex
    @inproceedings{zhou2021_informer,
      author    = {Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
      title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
      booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
      volume    = {35},
      number    = {12},
      pages     = {11106--11115},
      year      = {2021},
      doi       = {10.1609/aaai.v35i12.17325}
    }
    ```

---

### [DISC-02] - WU et al., 2021 (Atribuição Falsa de Base e Causa de Desempenho)
- **Localização no TCC:** Página 51 (Resultados do Track B, [resultados.tex L163](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L163)) e Página 32 (Trabalhos Correlatos, [fundamentacao.tex L355](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L355)).
- **Trecho Citado no TCC:**
  > "Essa estabilidade de resultados em torno de 76\% a 78\% de $F_1$-macro aponta para um teto prático nos dados de DGA estáticos baseados exclusivamente em cinco gases. Essa constatação elucida a aparente discrepância com estudos prévios da literatura, a exemplo de \citeonline{wu2021}, que reportavam acurácias superiores a 99\% sobre essa mesma base. Na versão original dos dados sem auditoria, a existência de 15,5\% de amostras redundantes e duplicatas distribuídas aleatoriamente entre treino e teste gerava vazamento espúrio de dados (\textit{data leakage}). Com o saneamento e a deduplicação mandatória conduzidos neste trabalho, o desempenho real sem viés situa-se na faixa de 76\% a 78\%, refletindo o verdadeiro teto físico da informação cromatográfica disponível." (p. 51, Seção 4.3)
- **Evidência no PDF Original (`artigos/fenrg-09-745744.pdf`):**
  > *"The DGA data set employed in this study is originated from Ref (Kirkbas et al., 2020). The data is divided into the training data set (125 samples) and the test data set (25 samples). These samples the number of samples used for the training and test process is shown in Table 2."* (Seção 2.3, p. 5 do PDF).
  > *"TABLE 6 \| Accuracy of different methods for the test samples... IGA-XGBoost: 99.2% (25 test samples)"* (p. 8 do PDF).
- **Problema Identificado:**
  1. Wu et al. **NÃO utilizaram a base chinesa** (2.321 amostras de Li et al. / Alan), mas sim uma base reduzida de apenas 150 amostras (125 de treino e 25 de teste) proveniente de Kirkbas et al. (2020).
  2. Afirmar categoricamente que a acurácia de 99,2% de Wu et al. decorreu de vazamento de duplicatas (15,5%) da base chinesa é factualmente falso. A alta acurácia do estudo de Wu et al. decorre do tamanho amostral ínfimo de teste (apenas 25 amostras) e de uma estrutura hierárquica em múltiplos estágios.
- **Ação Corretiva Proposta:**
  - **Texto sugerido para substituição no LaTeX ([resultados.tex L163](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L163)):**
    > "Essa estabilidade de resultados em torno de 76\% a 78\% de $F_1$-macro aponta para um teto prático em bases estáticas de DGA de larga escala baseadas exclusivamente nos cinco gases convencionais. Essa constatação contrasta com índices de acurácia superiores a 99\% frequentemente reportados na literatura, a exemplo de \citeonline{wu2021}, cujos experimentos foram conduzidos sobre conjuntos amostrais reduzidos (125 amostras de treino e 25 de teste). Além disso, uma hipótese plausível para explicar métricas infladas em estudos empíricos é a ausência de auditoria de duplicatas: em bases como a do Track B, a presença de 15,5\% de registros repetidos geraria vazamento espúrio de dados caso a divisão entre treino e teste ocorresse sem deduplicação prévia."

---

### [DISC-03] - WANG, LITTLER e LIU, 2023 (Erro de Escopo Metodológico e Métricas)
- **Localização no TCC:** Página 32 (Trabalhos Correlatos, [fundamentacao.tex L357](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L357)).
- **Trecho Citado no TCC:**
  > "Por outro lado, o trabalho de \citeonline{wang2023} focou na modelagem temporal com redes LSTM para projetar concentrações de gases combustíveis, reportando coeficientes de determinação elevados ($R^2 > 0,90$). No entanto, os autores utilizaram dados proprietários de transformadores em operação contínua e não avaliaram o impacto de acoplar as predições a classificadores diagnósticos de falha." (p. 32, Seção 2.8)
- **Evidência no PDF Original (`artigos/Accepted_manuscript_LSTM_Pure.pdf`):**
  > *"In this paper, a dynamic fault prediction approach is proposed using a long short-term memory (LSTM) model with intelligent classification to determine the running state of a transformer... Four artificial intelligence diagnostic models (support vector machine (SVM), K-nearest neighbors (KNN), decision tree, and artificial neural network (ANN)) were rendered and used for comparative fault prediction assessment... asserts that the LSTM-KNN model provides the highest and most reliable prediction accuracy"* (Abstract, p. 1 do PDF).
  > *"TABLE III: MAE, RMSE AND MSE OF FIVE PREDICTIVE MODELS"* (p. 6 do PDF – o paper não calcula $R^2$).
  > *"TABLE IX: FAULT PREDICTION ACCURACY OF THE CLASSIFICATION MODELS... LSTM-KNN: 97.6%"* (p. 8 do PDF).
- **Problema Identificado:**
  1. O artigo de Wang et al. **não utiliza o coeficiente $R^2$**; avalia a previsão por MAE, RMSE e MSE.
  2. O artigo **avalia sim o acoplamento da LSTM com 4 classificadores** (SVM, KNN, Árvore de Decisão e ANN). O que os autores **não fizeram** foi um estudo de ablação controlado com testes de significância estatística (bootstrap pareado) comparando contra uma linha de base estática (sem LSTM), impedindo comprovar se a LSTM agregou ganho diagnóstico real ou apenas atuou de forma redundante.
- **Ação Corretiva Proposta:**
  - **Texto sugerido para substituição no LaTeX ([fundamentacao.tex L357](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L357)):**
    > "Por outro lado, o estudo de \citeonline{wang2023} propôs uma abordagem híbrida acoplando a previsão temporal por LSTM a classificadores diagnósticos convencionais (como KNN e SVM), reportando baixos erros médios de projeção (MAE e RMSE) e acurácia de classificação de 97,6\%. Contudo, o trabalho baseou-se em dados proprietários contínuos e não realizou testes estatísticos pareados nem estudos de ablação controlados frente a classificadores puramente estáticos, deixando em aberto se a previsão temporal efetivamente adiciona ganhos diagnósticos significativos em relação às leituras pontuais já observadas."

---

### [DISC-04] - LI, WANG e SONG, 2019 (Atribuição Não Verificada de Laudos e Desempenho)
- **Localização no TCC:** Página 36–37 (Seção 3.2.2, [metodologia.tex L33](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/metodologia.tex#L33) e [L42](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/metodologia.tex#L42)).
- **Trecho Citado no TCC:**
  > "Cabe ressaltar que os diagnósticos da base foram atribuídos originalmente a partir de laudos operacionais e relatórios de inspeção física de concessionárias de energia. Como refletem laudos de campo e não ensaios controlados de laboratório com falhas puras, a base apresenta naturalmente sobreposições físico-químicas e casos limítrofes entre categorias... Essa complexidade intrínseca dos dados reais foi a motivação central para que \citeonline{li2019_membership} formulassem o diagnóstico via graus de pertinência difusos (\textit{membership degree}), justificando igualmente o teto prático de desempenho em torno de 76\% a 78\% obtido pelos modelos de aprendizado de máquina." (p. 36–37, Seção 3.2.2)
- **Evidência no Artigo Original (IEEE Access, 2019, DOI: 10.1109/ACCESS.2019.2903525):**
  > Li et al. (2019) propõem graus de pertinência difusa para lidar com a imprecisão de limites entre gases em transformadores. Porém, não há comprovação textual imediata no repositório de que os laudos derivaram de relatórios de inspeção física interna específicos, e a extrapolação de que isso "justifica igualmente o teto de 76% a 78%" foi uma inferência do TCC, pois Li et al. não testaram XGBoost nem reportaram essa faixa de F1-macro.
- **Problema Identificado:** Afirmação de procedência física não verificada documentalmente e atribuição indevida da justificativa da métrica do TCC ao trabalho de Li et al.
- **Ação Corretiva Proposta:**
  - **Texto sugerido para substituição no LaTeX ([metodologia.tex L42](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/metodologia.tex#L42)):**
    > "Os rótulos diagnósticos da base foram adotados conforme catalogados originalmente no repositório público. Como refletem registros operacionais de campo e não ensaios sintéticos de laboratório, os dados exibem sobreposições físico-químicas naturais e casos limítrofes entre categorias --- por exemplo, na classe de falhas térmicas severas (T3), a mediana observada para a razão $\text{C}_2\text{H}_4/\text{C}_2\text{H}_6$ é de apenas 0,25, distante do patamar teórico de 4,0 estabelecido pela norma IEC 60599. Essa dispersão inerente a dados reais motivou \citeonline{li2019_membership} a proporem a modelagem por graus de pertinência difusos (\textit{membership degree}), consistindo também em fator determinante para a estabilização empírica das métricas dos classificadores em torno de 76\% a 78\%."

---

### [DISC-05] - IBRAHIM, GHONEIM e TAHA, 2018 & ALAN, 2021 (Confusão de Proveniência do Dataset)
- **Localização no TCC:** Página 36 (Seção 3.2.2, [metodologia.tex L33](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/metodologia.tex#L33)).
- **Trecho Citado no TCC:**
  > "O Track B utiliza uma compilação de dados estáticos de ensaios laboratoriais de DGA obtidos em transformadores de potência em serviço na rede elétrica chinesa, disponibilizados publicamente por \citeonline{li2019_membership} via IEEE DataPort e catalogados na biblioteca de ferramentas de diagnóstico DGALab \cite{ibrahim2018_dgalab}, hospedada no repositório de livre acesso de \citeonline{alan2021_dataset}." (p. 36)
- **Evidência no PDF Original (`IET Generation Trans Dist - 2018 - Ibrahim...pdf`):**
  > DGALab é um software em MATLAB publicado em 2018 para implementar métodos normativos de DGA. O repositório GitHub de `alan-456` apenas agregou códigos e dados públicos (incluindo rotinas do DGALab e a planilha de Li et al., 2019). DGALab não é uma "base de dados" e não foi criado por Alan.
- **Problema Identificado:** Redação confusa que mistura a autoria da ferramenta computacional (Ibrahim et al., 2018), a fonte dos dados (Li et al., 2019 via IEEE DataPort) e o repositório de agregação no GitHub (`alan-456`).
- **Ação Corretiva Proposta:**
  - **Texto sugerido para substituição no LaTeX ([metodologia.tex L33](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/metodologia.tex#L33)):**
    > "O Track B adota uma base de dados estáticos de ensaios cromatográficos de transformadores em serviço, disponibilizada originalmente por \citeonline{li2019_membership} no IEEE DataPort e reunida com rotinas de diagnóstico da ferramenta DGALab \cite{ibrahim2018_dgalab} no repositório público mantido por \citeonline{alan2021_dataset}."
  - **Ajuste no `referencias.bib`:**
    Corrigir o autor de `alan2021_dataset` para evitar o nome inventado "Alan, C.":
    ```bibtex
    @misc{alan2021_dataset,
      author       = {{ALAN-456}},
      title        = {Transformer Fault Dataset},
      year         = {2021},
      publisher    = {GitHub},
      howpublished = {\url{https://github.com/alan-456/transformer-fault-dataset}},
      note         = {Acesso em: 20 out. 2026}
    }
    ```

---

### [DISC-06] - TIGHTIZ et al., 2020 (Atribuição Imprópria para Princípio Físico Básico)
- **Localização no TCC:** Página 17 (Seção 2.1, [fundamentacao.tex L9](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L9)).
- **Trecho Citado no TCC:**
  > "O funcionamento de um transformador baseia-se na Lei de Faraday, segundo a qual uma variação de fluxo magnético em um circuito induz uma força eletromotriz em outro circuito acoplado magneticamente \cite{tightiz2020}." (p. 17)
- **Problema Identificado:** Citação de um artigo especializado de 2020 sobre ANFIS/regras de associação para respaldar a Lei da Indução de Faraday (descoberta em 1831). A citação de Tightiz na introdução (p. 12) sobre a importância da manutenção é adequada, mas a atribuição da Lei de Faraday a esse artigo é tecnicamente imprópria.
- **Ação Corretiva Proposta:**
  - **Texto sugerido no LaTeX ([fundamentacao.tex L9](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L9)):**
    Remover a citação `\cite{tightiz2020}` do final da frase da Lei de Faraday ou substituí-la por livro-texto clássico de máquinas elétricas, mantendo `tightiz2020` apenas onde discute sistemas inteligentes de diagnóstico.

---

## 3. Plano de Ação Sequencial de Correção

Checklist ordenado por prioridade de impacto para o autor executar diretamente no manuscrito:

- [x] **Prioridade 1 (Crítica):**
  - [x] **Eliminar a referência alucinada `Zhu et al. (2023)`**: substituir a entrada no `.bib` por `zhou2021_informer` (Zhou et al., 2021 - Informer, AAAI 2021), e atualizar as 4 ocorrências no texto ([fundamentacao.tex L129, L166, L346 e L361](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L129)).
  - [x] **Corrigir a afirmação sobre Wu et al. (2021) em 4.3 ([resultados.tex L163](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/resultados.tex#L163))**: remover a alegação de que Wu et al. usaram a mesma base e de que seus >99% decorreram de 15,5% de duplicatas, reescrevendo como comparação entre bases de grande porte e amostras reduzidas (150 amostras).

- [x] **Prioridade 2 (Alta):**
  - [x] **Adequar o escopo do trabalho de Wang, Littler & Liu (2023) ([fundamentacao.tex L357](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L357))**: retirar a menção a $R^2 > 0,90$ (que inexiste no paper) e corrigir a afirmação de que não acoplaram a classificadores, pontuando com precisão que faltou ao estudo de Wang um protocolo controlado de ablação com testes estatísticos pareados frente a modelos puramente estáticos.
  - [x] **Suavizar as afirmações sobre a base de Li et al. (2019) ([metodologia.tex L42](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/metodologia.tex#L42))**: não fazer afirmações categóricas sobre relatórios de inspeção física interna nem atribuir a Li et al. a justificação do teto de 76–78% obtido pelo classificador do TCC.

- [x] **Prioridade 3 (Média):**
  - [x] **Esclarecer a procedência do Track B ([metodologia.tex L33](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/metodologia.tex#L33))**: distinguir claramente a planilha de dados de Li et al. (2019), o software DGALab de Ibrahim et al. (2018) e o repositório de agregação de `alan-456` no GitHub.
  - [x] **Ajustar a entrada BibTeX de `alan2021_dataset` ([referencias.bib L333](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/referencias.bib#L333))**: substituir o autor inventado "Alan, C." pelo identificador institucional ou handle do repositório `{ALAN-456}`.
  - [x] **Desvincular a citação de Tightiz et al. (2020) da Lei de Faraday ([fundamentacao.tex L9](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex#L9))**.
