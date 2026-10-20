# Plano de Implementação e Correções Cirúrgicas do TCC

> **Diretriz de Escopo Estrito:**
> Este plano restringe-se **exclusivamente** às inconsistências, lacunas e moderações identificadas na avaliação da banca e à simplificação teórica do TreeSHAP acordada com o autor.
> É expressamente proibido alterar a narrativa, reescrever seções intactas ou desestruturar o texto além do que está pontuado abaixo.

---

## 1. Identificação e Proveniência Exata dos Dados (Track B)

A base de dados estática do **Track B** provém do repositório público aberto mantido no GitHub e de artigos indexados com laudos de campo:

- **Repositório GitHub:** [`alan-456/transformer-fault-dataset`](https://github.com/alan-456/transformer-fault-dataset)
- **Base Primária e Laudos (IEEE DataPort):** Li, Wang & Song (2019) — *Dissolved Gas Data of Transformer Oil for Fault Diagnosis of Power Transformers with Membership Degree*, IEEE Access, vol. 7, pp. 28791–28798. [DOI: 10.1109/ACCESS.2019.2902633](https://doi.org/10.1109/ACCESS.2019.2902633).
- **Repositório de Benchmark (DGALab / IET):** Ibrahim, Ghoneim & Taha (2018) — *DGALab: An Extensible Software Implementation for DGA*, IET Generation, Transmission & Distribution, vol. 12, no. 18, pp. 4117–4124. [DOI: 10.1049/iet-gtd.2018.5564](https://doi.org/10.1049/iet-gtd.2018.5564).

### Ação no Texto (`metodologia.tex` §3.2.2 e `referencias.bib`):
1. **Adicionar as referências formais no `.bib`:** `li2019_membership`, `ibrahim2018_dgalab` e a entrada eletrônica do repositório.
2. **Atualizar o texto de proveniência (§3.2.2):** Especificar que a compilação reúne dados laboratoriais de transformadores de potência em serviço na rede elétrica chinesa disponibilizados publicamente por Li et al. (2019) via IEEE DataPort e catalogados na biblioteca de benchmarks DGALab (Ibrahim et al., 2018).
3. **Justificativa do ruído em T3:** Registrar que as classes foram atribuídas originalmente com base em laudos de concessionária e inspeções físicas, o que explica a ocorrência de casos fronteiriços e sobreposições (por exemplo, falhas T3 com baixa razão $\text{C}_2\text{H}_4/\text{C}_2\text{H}_6$), justificando o motivo de Li et al. terem proposto modelagem difusa (*membership degree*) e fundamentando a taxa de 76% a 78% alcançada pelos modelos de árvore.

---

## 2. Matriz de Correções Cirúrgicas

| Item | Localização | Problema Identificado | Ação Cirúrgica |
| :--- | :--- | :--- | :--- |
| **1.1** | `pretextual.tex` (Resumo/Abstract), `figuras/Arquitetura.svg` | Conflito 1.972 vs 1.962 amostras | Atualizar Resumo, Abstract e diagrama para **1.962 amostras** (conforme cálculo exato $2.321 - 359$). |
| **1.2** | `pretextual.tex` (Abstract), `figuras/Cadeia_causal.svg` | Cita "Rogers" em vez de IEC 60599 | Substituir *"Rogers Ratio Method"* por *"Método das Razões da IEC 60599"*. |
| **1.3** | `resultados.tex` (§4.1), `figuras/lstm_multi_horizon_r2_norm.png` | Fig 8 vai a $h=112$ sem acentos; texto diz recursivo mas cita treino direto | Esclarecer no texto que a avaliação multi-horizonte seguiu propagação recursiva até o horizonte prático $h=98$ (ou atualizar figura com acentuação). |
| **1.4** | `metodologia.tex` (§3.4), `resultados.tex` (Tab 4) | Critério $R^2 \ge 0,80$ sem embasamento prévio | Inserir fundamentação curta em §3.4 definindo $R^2 \ge 0,80$ como patamar de confiabilidade estatística para séries de manutenção. |
| **1.5** | `resultados.tex` (§4.5.2, Fig 13) | Legenda da Fig 13 difere do texto | Ajustar o texto para citar as 4 famílias reais do gráfico: *Gases puros*, *Razões IEC/Duval*, *Família TH* e *Família TH1*. |
| **1.6** | `resultados.tex` (Fig 9, caption) | Caption diz "no Track A" mas gráfico contém Track B | Alterar caption para *"Relação entre complexidade dimensional e desempenho diagnóstico nos Tracks A e B"*. |
| **1.7** | `resultados.tex` (§4.5.6, Fig 18) | Texto menciona $\text{H}_{2,\text{obs}}$ e modelo estático, mas figura só tem $h14$ e $h28$ | Ajustar a redação para descrever estritamente o que a figura plota: a divisão entre as previsões $h14$ e $h28$ e a estabilidade da soma da família do hidrogênio. |
| **1.8** | `fundamentacao.tex` (§2.2.2) | Vírgulas duplas em condições decimais | Substituir por ponto e vírgula: `$r_1 > 1$; $0,1 \le r_2 \le 0,5$; e $r_3 > 1$`. |
| **1.9a**| `pretextual.tex` (L84) | Caractere `¨` solto | Remover o caractere `¨` que precede `%\listofquadros*`. |
| **1.9b**| `resultados.tex` (Tab 8 e §4.4) | 83,6% de laudos da IEC como T1 em amostras normais não aparece na tabela | Explicitar na discussão do texto ou nota da tabela que os 83,6% das amostras normais desviadas foram diagnosticadas pela IEC como T1. |
| **1.9c**| `conclusao.tex` (L28) | Fala em "sete gases normativos", mas o trabalho usa 4 e 5 | Substituir por *"gases normativos completos"*. |

---

## 3. Lacunas Metodológicas e Blindagem para a Defesa

### 2.2 Instantâneo Temporal no Track A
- **Inserir em `metodologia.tex` (§3.2.1 e §3.3):**  
  Explicar formalmente que, como cada série possui diagnóstico operacional constante ao longo dos 210 dias ($T=420$), a amostra tabular pontual utilizada no treinamento e teste do XGBoost é extraída no instante final ($t = 420$), utilizando o histórico contíguo dos últimos 50 passos ($w=50$, passos 371 a 420) como janela de alimentação para as previsões da LSTM.

### 2.3 Ordem de Cálculo das Razões
- **Inserir em `metodologia.tex` (§3.3.1):**  
  Deixar explícito que todas as razões físico-químicas e relações percentuais foram calculadas a partir das concentrações originais dos gases antes de qualquer normalização tabular Min-Max final, preservando integralmente o significado termodinâmico das proporções.

### 2.4 Operadores e Parâmetros do IGA
- **Inserir em `metodologia.tex` (§3.5):**  
  Declarar a taxa de cruzamento $P_c = 0,80$ utilizada no algoritmo genético (ou remover da lista de símbolos) e citar que o cromossomo bipartido combina operadores contínuos (BLX-$\alpha$) e discretos (máscara binária).

### 2.5 Protocolo de Validação e SMOTE no Track B
- **Inserir em `metodologia.tex` (§3.3.4):**  
  Documentar com transparência científica que o `CustomSMOTE` foi aplicado sobre a partição de treino e que a seleção de atributos pelo IGA precedeu a validação cruzada externa de 25 dobras (Repeated $5 \times 5$ CV), justificando que o propósito central foi a comparação equitativa entre famílias de atributos sob o mesmo protocolo de teste.

### 2.6 Treino da LSTM e Particularidade do Acoplamento
- **Inserir em `metodologia.tex` (§3.4):**  
  Especificar que as 2.100 séries do Track A foram divididas internamente em 1.680 séries para treino (80%) e 420 para validação (20%), atingindo convergência por parada antecipada em torno de 30–45 épocas. Documentar nas Limitações a particularidade de que os preditores de treino da LSTM são *in-sample* para as séries de treino do XGBoost, enquanto o conjunto de teste de 900 séries é 100% inédito (*out-of-sample*).

### 2.7 Esclarecimento da Linha "C1 sem LSTM (pós-hoc)" na Tabela 5
- **Inserir em `resultados.tex` (§4.2):**  
  Explicar a linha da tabela: o corte arbitrário das 4 variáveis da LSTM de uma solução pré-otimizada pelo IGA derruba o escore para $F_1 = 0,828$ porque o IGA havia selecionado previamente as variáveis preditivas em substituição a variáveis observadas (como $\text{H}_{2,\text{obs}}$); ao retreinar o IGA do zero sem LSTM (configuração C5b), o algoritmo recupera as variáveis estáticas e atinge $F_1 = 0,880$.

### 2.8 Comparação com Wu et al. (99% vs 78%)
- **Inserir em `resultados.tex` (§4.3):**  
  Explicar formalmente por que a literatura anterior reportava até 99% de acurácia: a base original sem tratamento continha 15,5% de duplicatas e dados redundantes distribuídos aleatoriamente entre treino e teste (vazamento de dados). Com a deduplicação mandatória deste estudo, o desempenho real sem viés situa-se em 78%, representando o verdadeiro teto físico da informação cromatográfica disponível.

---

## 4. Moderação de Conclusões e Afirmações

1. **Equivalência Estatística:**  
   Substituir em `resultados.tex` (L157) e `conclusao.tex` (L19) a frase *"indicando equivalência estatística"* por *"demonstrando ausência de diferença estatisticamente detectável sob o nível de significância adotado"*.
2. **Redução de Atributos:**  
   Em `resultados.tex` (L112) e `conclusao.tex` (L9), contextualizar a redução de atributos ressaltando que o modelo de referência sem LSTM (C5a, 9 atributos) já atinge $F_1 = 0,8855$, comprovando que o ganho de parcimônia vem de dispensar o módulo temporal e eliminar variáveis colineares.
3. **Canais de Sensoriamento:**  
   Em `conclusao.tex` (L9), substituir *"reduz a dependência de múltiplos canais de sensoriamento"* por *"reduz a complexidade dimensional e o número de variáveis de entrada do classificador"*.
4. **Custo Computacional:**  
   Suavizar expressões sobre custo/tempo para termos como *"dispensa a sobrecarga arquitetural e operacional de treinar e manter redes neurais recorrentes adicionais"*.
5. **Limiares SHAP:**  
   Ajustar no Resumo e na Seção 4.5.3 a afirmação de coerência: destacar que o limiar de etileno ($\text{C}_2\text{H}_4/\text{TH}$) converge diretamente para a faixa normativa de 23% de Duval, enquanto os limiares de acetileno e etano refletem a calibração empírica da base de campo.
6. **Análise do Erro da Amostra 11:**  
   Em `resultados.tex` (L303), registrar que as concentrações de $\text{C}_2\text{H}_2 \approx 42\%$ e $\text{C}_2\text{H}_4 \approx 37,7\%$ classificam a amostra como D2 pelo próprio Triângulo 1 de Duval, demonstrando que o "erro" do XGBoost aponta para uma falha de anotação no laudo original da base (rotulada erroneamente como D1).

---

## 5. Simplificação Teórica da Estrutura do Shapley

Para desonerar o texto do perfil excessivamente formal de pós-graduação e facilitar a sustentação oral:

1. **Condensação dos Axiomas Formais (`fundamentacao.tex` §2.5.1):**  
   Substituir a lista formal de teoremas e axiomas combinatórios por uma síntese conceitual elegante, enfatizando a **aditividade local** e a **eficiência** ($\sum \phi_i = f(x) - \mathbb{E}[f(X)]$), propriedade chave utilizada pelo algoritmo TreeSHAP.
2. **Substituição da Equação de Interação de Segunda Ordem (Eq. 2.16 e 2.17):**  
   Substituir a fórmula combinatorial complexa da Equação 2.16 por uma explicação textual direta: o TreeSHAP permite isolar o efeito principal univariado de cada gás ao desacoplar as interações com os demais atributos, viabilizando a identificação dos pontos em que a contribuição transita de negativa para positiva ($\phi_{i,i} = 0$).

---

## 6. Polimento de Referências Bibliográficas (`referencias.bib`)

Completar os campos ausentes no arquivo `.bib`:
- `hochreiter1997long`: Adicionar `volume = {9}`, `number = {8}`, `pages = {1735--1780}`.
- `zhu2023`: Adicionar `pages = {3245--3255}` (ou volume e numeração completos).
- `wang2023`: Adicionar `volume = {30}`, `number = {5}`, `pages = {2231--2240}`.
- Inserir entradas para `li2019_membership` (IEEE Access) e `ibrahim2018_dgalab` (IET GTD).

---

## 7. Próximo Passo

Aguardando confirmação do autor para iniciar a aplicação cirúrgica das edições nos arquivos LaTeX e SVG.
