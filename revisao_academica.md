# Revisão Acadêmica Crítica da Defesa e Nota Técnica sobre o Modelo PowerPoint

**Avaliação realizada sob a ótica de Banca Examinadora em Engenharia da Computação**  
**Trabalho avaliado:** *Predição Dinâmica de Falhas em Transformadores: Abordagem Híbrida de Previsão Temporal (LSTM) e Classificação Otimizada (IGA-XGBoost)* — Ândrio Gabriel Epping (UNISINOS, 2026).

---

## 1. Análise Crítica pelos 6 Critérios da Banca Examinadora

### 1.1 Clareza e Foco do Problema (Nota de Banca: Excelente)
- **Pontos fortes:** O Slide 3 delimita objetivamente três lacunas reais e verificáveis na literatura DGA: (i) ausência da categoria Normal e fronteiras rígidas em Duval/IEC 60599; (ii) inflação métrica em bases pequenas (`N=25` no teste de Wu et al., 2021) ou sem deduplicação; e (iii) avaliação desacoplada de modelos LSTM sem ablação sobre o classificador (Wang et al., 2023). A pergunta científica está formulada sem presunção de superioridade.
- **Ponto de atenção na fala oral:** Evitar que a banca interprete os dois tracks como "dois TCCs diferentes". Deixar cristalino logo nos Slides 4–5 que o Track A isola a hipótese temporal (série contínua, 4 gases, 4 classes) e o Track B isola o benchmark normativo completo (5 gases, 7/6 classes).

### 1.2 Coerência Metodológica e Controle de Vazamento (Nota de Banca: Muito Forte)
- **Pontos fortes:** A auditoria que expurgou 359 registros (15,5%: 303 duplicatas idênticas e 56 conflitos de rótulo) no Track B, o isolamento estrito de `ScalerPerGas` / `MinMaxScaler` no treino, o cálculo das razões molares em ppm bruto **antes** da transformação `ln(x+1)` e o `CustomSMOTE` encapsulado em `imblearn.Pipeline` demonstram domínio raro de engenharia de experimentos em graduação.
- **Fragilidade potencial que a banca poderá arguir:**
  1. *Previsões LSTM de treino no Track A:* As projeções `h=14` e `h=28` usadas para treinar o XGBoost (nas 1.785 séries de treino) vieram de uma LSTM ajustada sobre essas mesmas séries (sem validação cruzada out-of-fold temporal), embora as 900 séries de teste sejam 100% cegas. A resposta correta (constante no Slide 14 e no roteiro) é que esse viés tornaria a LSTM **otimista**, e mesmo assim ela não superou `C5a` no teste (`p=0,238`).
  2. *CV 5×5 pós-seleção no Track B:* A validação cruzada repetida 5×5 (`fase56.py`) rodou com o conjunto de atributos e hiperparâmetros já selecionados pelo IGA por custo computacional, sem re-otimização genética aninhada (Nested CV), o que é explicitamente admitido no Slide 14.

### 1.3 Rigor Técnico na Explicação dos Algoritmos (Nota de Banca: Completo e Fiel ao Código)
- **Pontos fortes:**
  - **LSTM (Slide 7 + Apoio A1):** Descreve literalmente `src/lstm/model.py` (2 camadas de 64 unidades, `return_sequences=True` na 1ª e `False` na 2ª com estado oculto `h_50 em R^64`, `Dropout(0,20)`, saída `Dense(4, linear)`), bem como os tamanhos exatos dos tensores `(660.450, 50, 4)` e o loop recursivo FIFO.
  - **IGA-XGBoost (Slide 8 + Apoio A2/A3):** Separa de modo exemplar o papel interno do XGBoost (`mlogloss` regularizada em 2ª ordem) do critério evolutivo (`F1-macro` em `StratifiedKFold(5)` no treino), e resolve antecipadamente a provável dúvida sobre `max_depth=12` mostrando como `colsample_bytree=0,3433`, `reg_lambda=3,9425`, `gamma=1,4356` e `min_child_weight=3` (com `h_{i,c} <= 0,25` exigindo >=12 amostras incertas por folha) controlam o sobreajuste.

### 1.4 Qualidade das Evidências Visuais (Nota de Banca: Alta)
- Os gráficos embutidos não têm função decorativa: cada figura sustenta um argumento específico da fala (e.g., `fig4_thresh_ratio_C2H4_TH_D1.png` prova a recuperação do corte de 23% de Duval; `fig7_lstm_credit_migration.png` comprova a troca de crédito entre `lstm_h14_H2` e `lstm_h28_H2` com soma constante em `0,4734`; `fig5_duval_triangle.png` localiza geometricamente os 43 erros do C3 nas bordas entre regiões).
- O diagrama `Cadeia_causal.svg` foi exportado em 300 DPI sem serrilhado, e o diagrama de arquitetura do Slide 5 foi reconstruído como formas vetoriais nativas editáveis no PowerPoint.

### 1.5 Interpretação Crítica dos Resultados (Nota de Banca: Ponto Alto do Trabalho)
- **Maior mérito científico do TCC:** Em vez de forçar a narrativa de que a abordagem híbrida "venceu", o trabalho demonstra com **baseline de persistência** (`R² = 0,9970` em 1 semana) e **bootstrap pareado** (`ΔF1 = +0,0056`, `p = 0,2384`) *por que* a LSTM não agregou ao diagnóstico do instante atual — atribuindo o fato à alta inércia das séries e à migração de crédito colinear medida via TreeSHAP (`0,473 ± 0,013`), além de explicar por que Duval/IEC pontuaram baixo (ausência da classe Normal e anomalia `C₂H₄/C₂H₆ = 0,25` em T3 na base compilada).
- **Cuidado editorial já aplicado:** Não cita `C1_menos_LSTM_posthoc` (`0,8280`) como prova de valor da LSTM; marca-o apenas como artefato pós-hoc e adota `C5a` (`0,8855`) e `C5b` (`0,8796`) como controles causais limpos. Além disso, acompanha o TreeSHAP de ressalva explícita de que atribuição de Shapley em log-odds reflete associação no modelo e não causalidade física direta.

### 1.6 Fluxo Lógico e Viabilidade em 20 Minutos (Nota de Banca: Adequado)
- Os 15 slides principais têm densidade informacional balanceada e distribuição de tempos exata (20:00), apoiados por 6 slides reservados (Apoio A1–A6) que retiram fórmulas longas, matrizes 7×7 e a tabela completa da auditoria normativa da exposição principal, mantendo-os prontos para a arguição.

---

## 2. Nota Técnica sobre o Modelo Institucional PowerPoint (`python-pptx` vs. OpenXML)

Durante a inspeção estrutural de [apt_modelo-institucional_2027.pptx](file:///c:/Users/ResTIC55/Documents/TCC_Final/apt_modelo-institucional_2027.pptx), verificaram-se duas restrições técnicas relevantes:

1. **Ausência de `python-pptx` no ambiente e limitações de `asvg:svgBlip`:**
   - Todos os 34 slides do modelo institucional utilizam exclusivamente o layout vazio `Blank` (`slideLayout7.xml`), incorporando as ondas gráficas de rodapé (`image11.png`), fundos institucionais (`image2.png`, `image37.png`, `image50.png`) e logotipos vetoriais (`image3.svg` branco e `image7.svg` azul UNISINOS via extensão `<asvg:svgBlip>` de drawingML 2016) diretamente no `spTree` de cada slide de exemplo, em vez de consolidá-los em layouts semânticos no `slideMaster`.
   - Adicionalmente, a biblioteca `python-pptx` não está pré-instalada nos interpretadores locais (ambiente isolado sem rede) e sua API de alto nível não expõe construtores nativos para clonar slides preservando extensões `asvg:svgBlip`.
2. **Solução adotada em [gerar_apresentacao.py](file:///c:/Users/ResTIC55/Documents/TCC_Final/gerar_apresentacao.py):**
   - Implementou-se um gerador OpenXML determinístico sobre `lxml.etree` e `zipfile` (utilizando `& "C:\Program Files\Inkscape\bin\python.exe"`), clonando diretamente a base OpenXML e os relacionamentos SVG/PNG dos slides-modelo `slide2` (capa), `slide13` (conteúdo), `slide23` (divisor) e `slide34` (encerramento), mantendo o tema `Trebuchet MS` e a paleta `#253786 / #21409A / #FE4900 / #00BAC8`, podando automaticamente as imagens não utilizadas de outros slides do modelo original (~28 MB → ~4,96 MB).
   - Todos os elementos da apresentação (caixas de texto, títulos, cartões, setas, conectores, tabelas e notas do apresentador em `notesSlides`) são objetos nativos **100% editáveis** no Microsoft PowerPoint.
   - **Limitação conhecida de tipografia (`Trebuchet MS`):** A fonte institucional instalada (`C:\Windows\Fonts\trebuc.ttf`) não possui glifos nativos para setas Unicode (`→`, `↔`), operador Hadamard (`⊙`), símbolo `∈`, `ℝ`, superíndice `⁻` e til combinante (`U+0303`), substituídos de forma consistente em [gerar_apresentacao.py](file:///c:/Users/ResTIC55/Documents/TCC_Final/gerar_apresentacao.py) por notações suportadas (`->`, `<->`, `*`, `em`, `R⁶⁴`, `10^-3`, `C~_t`), eliminando caixas de substituição `□`.

---

## 3. Sabatina Antecipada — 8 Perguntas Difíceis da Banca e Respostas Diretas

1. **"Se a LSTM não melhorou a classificação do transformador, o seu TCC fracassou na hipótese principal?"**
   - *Resposta:* Em ciência e engenharia, refutar uma hipótese por meio de ablação limpa (`C2 vs. C5a`, `p=0,238`) e explicar o mecanismo (alta inércia temporal com `R²_pers = 0,9970` e migração colinear de crédito em `H₂` somando `0,4734`) constitui uma contribuição científica sólida — evita que a indústria implemente e mantenha redes recorrentes caras onde modelos tabulares estáticos entregam o mesmo diagnóstico (`F1=0,886`).
2. **"Por que usar um algoritmo genético pesado (IGA) em vez de Grid Search, Random Search ou Optuna/BOHB?"**
   - *Resposta:* O IGA resolve simultaneamente o espaço misto de seleção binária de variáveis (17 ou 19 bits) e sintonia contínua (9 hiperparâmetros), seguindo e estendendo a linha de Wu et al. (2021) para possibilitar comparação metodológica direta (`M3`). A comparação quantitativa direta com Random/Bayesian Search foi formalmente registrada nas limitações e trabalhos futuros.
3. **"Por que não usar SMOTE padrão nas 19 colunas do Track B em vez de criar o CustomSMOTE?"**
   - *Resposta:* Porque as 14 variáveis derivadas são funções não lineares das 5 concentrações brutas (`(a₁+a₂)/(b₁+b₂) ≠ ½(a₁/b₁ + a₂/b₂)`). Interpolar razões diretamente gera assinaturas gasosas fisicamente impossíveis; o `CustomSMOTE` sintetiza somente nos 5 gases e recalcula exatamente as 14 razões/percentuais no espaço físico.
4. **"Como explicar que `C₂H₄/C₂H₆` apresentou mediana `0,25` em T3 quando a norma IEC 60599 exige `> 4,0`?"**
   - *Resposta:* Nossa auditoria normativa (`N=1.962`, Slide 22) demonstrou que na base pública compilada (Li et al., 2019 / DGALab) grande parte dos casos rotulados historicamente como T3 possui perfil de gases mais brando que o critério estrito da IEC, caindo em T2/DT na projeção física. O XGBoost alcança ~0,80 aprendendo a convenção de rotulagem da própria base.
5. **"O valor `max_depth=12` escolhido pelo IGA tanto no Track A quanto no Track B não indica memorização dos dados?"**
   - *Resposta:* Não, porque o IGA associou `max_depth=12` a forte regularização no Track B (Slide 20): `colsample_bytree=0,3433` restringe cada árvore a ~5 atributos aleatórios, `reg_lambda=3,9425` amortece pesos foliares, `gamma=1,4356` exige ganho alto de redução de perda e `min_child_weight=3` impõe >=12 amostras incertas por folha.
6. **"No Slide 12, por que `C1` obteve `F1=0,8909` e `C1 sem LSTM pós-hoc` caiu para `0,8280`?"**
   - *Resposta:* Porque os 8 atributos de C1 continham `lstm_h14_H2`, `lstm_h28_H2`, `lstm_h28_C2H4` e `lstm_h28_C2H2` em substituição a `H₂`, `C₂H₄` e `C₂H₂` brutos. Removê-los depois deixa C1 cego a três gases essenciais. Quando os mesmos atributos são avaliados sem LSTM desde o início (`C5a = 0,8855`, `C5b = 0,8796`), a diferença para `C1`/`C2` desaparece (`p = 0,238` / `0,331`).
7. **"O bootstrap estratificado garante validade fora dessa base?"**
   - *Resposta:* O bootstrap pareado estima a variabilidade amostral condicionada à população de teste representada pelos `N=900` (Katser) e `N=295` (Track B) registros. Generalização externa adicional exige validar em séries de campo com os 7 gases completos e transições reais de falha (Slide 15).
8. **"Se `R²` da LSTM supera 0,99 em `h=28`, onde essa rede realmente tem aplicação prática em subestações?"**
   - *Resposta:* Em **prognóstico de tendências e Vida Útil Remanescente (RUL)** — estimar quantos dias ou semanas faltam para um transformador sob evolução lenta ultrapassar limiares de alarme normativos da IEEE C57.104 / IEC 60599, tarefa em que prever o valor futuro exato da trajetória (`R² >= 0,80` até 7 semanas) é diretamente acionável, ao contrário da classificação do estado presente.
