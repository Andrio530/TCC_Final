# Plano de Ajuste: Figuras 8 e 18 e Citação de Transformers (`zhu2023`)

**Data:** 04 de novembro de 2026  
**Objetivo:** Diagnosticar e planejar a correção visual das Figuras 8 e 18 e a regularização bibliográfica da citação `zhu2023` (Transformers em séries temporais).  
**Status:** Executado e Verificado com Sucesso (Opção A aplicada).

---

## 1. Diagnóstico e Plano para a Figura 18 (`fig7_lstm_credit_migration.png`)

### 1.1. Causa Raiz dos Problemas Visuais
Ao inspecionar o arquivo [`tcc-transformer-fault/src/evaluation/fase6_shap_track_a.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-transformer-fault/src/evaluation/fase6_shap_track_a.py#L79-L115) e a imagem gerada em [`tcc-unisinos/figuras/fase6/fig7_lstm_credit_migration.png`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/figuras/fase6/fig7_lstm_credit_migration.png), identificaram-se dois conflitos geométricos:

1. **Colisão entre Legenda e Título:**
   * O título foi configurado com espaçamento `pad=28` e a legenda foi posicionada fora do eixo em `bbox_to_anchor=(0.5, 1.12)`. Como ambos ocupam a mesma cota vertical acima da borda superior do gráfico, a caixa da legenda ficou sobreposta exatamente em cima do texto do título.
2. **Colisão entre os Valores da Soma Média (`|SHAP|`) e as Porcentagens de Crédito (`%`):**
   * O eixo esquerdo (`ax`, barras empilhadas de `0%` a `100%`) estava com limite `ylim(0, 115)`. As porcentagens da barra superior salmão (`34.5%`, `32.3%`, `31.9%`, `35.9%`, `36.1%`) ficam posicionadas no centro da faixa salmão, na altura equivalente a `y1 ≈ 82%–84%`.
   * O eixo direito (`ax2`, linha preta da Soma Média de `|SHAP|` variando de `0.4624` a `0.4964`) estava com limite `ylim(0, max(soma) * 1.35)` $\approx (0, 0.67)$. Nessa escala, o valor `0.47` fica a `70%` da altura vertical da figura — que corresponde exatamente a `0.70 × 115 = 80.5%` no eixo esquerdo!
   * Assim, a linha preta atravessou o meio da barra salmão e os rótulos `0.4798`, `0.4650`, `0.4964`, `0.4635` e `0.4624` caíram exatamente em cima dos textos `34.5%`, `32.3%`, `31.9%`, `35.9%` e `36.1%`.

### 1.2. Solução Técnica Planejada
No script [`fase6_shap_track_a.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-transformer-fault/src/evaluation/fase6_shap_track_a.py):
1. **Escalonamento Coordenado dos Eixos Y (Desacoplamento Visual):**
   * Configurar o eixo primário para `ax.set_ylim(0, 142)` mantendo as marcações visíveis restritas a `ax.set_yticks([0, 20, 40, 60, 80, 100])`. Com isso, o topo das barras (`100%`) ocupa os `70%` inferiores da altura do gráfico, e os textos `34.5%` a `36.1%` ficam em `~58%` da altura total, **isolados dentro da barra salmão**.
   * Configurar o eixo secundário para `ax2.set_ylim(0, 0.62)`. Nessa proporção, os pontos da linha preta (`0.4624` a `0.4964`) passam a ocupar `~75% a 80%` da altura total da figura (equivalente à cota `y1 ≈ 107–113`, ou seja, **na área branca limpa logo acima do topo das barras de 100%**).
   * Os rótulos da soma média (`0.4798`, etc.) ficarão posicionados logo acima dos marcadores pretos (`xytext=(0, 7)`), sem qualquer contato com as barras ou com as porcentagens.
2. **Legenda no Canto Direito Superior e Título Livre:**
   * Posicionar a legenda no **canto direito superior interno** (`loc='upper right'`, `ncol=1` ou `ncol=2` compacto, `frameon=True`, `framealpha=0.95`), ocupando a faixa superior direita (`y1 ≈ 118–140`) acima da linha preta.
   * Reduzir o `pad` do título de `28` para `12`, deixando o título no topo externo totalmente livre.

---

## 2. Diagnóstico e Plano para a Figura 8 (`lstm_multi_horizon_r2_norm.png`)

### 2.1. Causa Raiz do Erro nas Semanas (Curvas parando em ~3,3 semanas)
Ao verificar o histórico de geração da imagem [`tcc-unisinos/figuras/lstm_multi_horizon_r2_norm.png`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/figuras/lstm_multi_horizon_r2_norm.png):
* Os 12 horizontes avaliados no arquivo [`lstm_multi_horizon_results_norm.json`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-transformer-fault/reports/metrics/lstm_multi_horizon_results_norm.json) são $h \in \{1, 2, 4, 7, 14, 28, 42, 56, 70, 84, 98, 112\}$ passos.
* Como cada passo equivale a **12 horas** (`HORAS_POR_PASSO = 12`), o horizonte $h=112$ corresponde a $112 \times 12 = 1.344\text{ horas}$ (**8,0 semanas**).
* Na geração anterior da figura, o script auxiliar multiplicou os passos por `5` (`horas = [h * 5 for h in hs]`, indo de `5h` até `560h`, o que equivale a apenas $560 / 168 = 3,33\text{ semanas}$), enquanto a linha vertical pontilhada verde utilizou o valor `hm_h = 1344` (`8,0 semanas`) salvo no JSON. Isso comprimiu as curvas no terço esquerdo do gráfico (até 3,3 semanas) e deixou um espaço em branco artificial até a semana 8.

### 2.2. Solução Técnica Planejada
No script [`tcc-transformer-fault/src/lstm/multi_horizon_eval.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-transformer-fault/src/lstm/multi_horizon_eval.py#L387-L453):
1. Utilizar o cálculo oficial `horas = [h * 12 for h in hs]` (`[12, 24, 48, 84, 168, 336, 504, 672, 840, 1008, 1176, 1344]` horas) para todas as curvas ($\text{H}_2$, $\text{CO}$, $\text{C}_2\text{H}_4$, $\text{C}_2\text{H}_2$ e $R^2$ médio).
2. Com isso, todas as curvas percorrerão de forma coerente todo o gráfico até **1.344 horas (8,0 semanas)**, coincidindo com a linha vertical de limite em $h=112$ (ou destacando $h=98$ / 7 semanas como último horizonte com todos os gases $\ge 0,80$ e $h=112$ / 8 semanas onde o $\text{H}_2$ cruza $0,80$).
3. Sincronizar o eixo X superior em semanas (`0` a `8` semanas) e regerar `lstm_multi_horizon_r2_norm.png` nas pastas `tcc-transformer-fault/reports/figures/` e `tcc-unisinos/figuras/`.

---

## 3. Diagnóstico e Recomendações para a Citação dos Transformers (`zhu2023`)

### 3.1. Por que apareceu `(?)` no PDF e onde está citado?
Na auditoria de referências realizada anteriormente ([`auditoria_referencias.md`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/auditoria_referencias.md#L31-L58)), constatou-se que a referência `Zhu et al. (2023)` não existia no periódico *IEEE Transactions on Power Systems*. Tratava-se de uma referência incorreta para o artigo real da arquitetura **Informer** (**Zhou et al., 2021** — *"Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"*, publicado na AAAI 2021, cujo PDF original `17325-13-20819-1-2-20210518.pdf` está salvo na pasta `tcc-unisinos/artigos/`).

No arquivo [`tcc-unisinos/referencias.bib`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/referencias.bib#L305-L314), a chave `@article{zhu2023}` já havia sido substituída por `@inproceedings{zhou2021_informer}` (e `@inproceedings{vaswani2017}`). Porém, no arquivo [`tcc-unisinos/fundamentacao.tex`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex), restaram **4 menções órfãs** a `zhu2023` / `Zhu et al. (2023)`:

1. **Linha 133 (Fonte da Tabela 1 – `tab:algoritmos_ml`):**  
   `\citeonline{zhu2023}` $\rightarrow$ renderiza como **`(?)`** no PDF.
2. **Linha 170 (Corpo da Tabela 1 – `tab:algoritmos_ml`, grupo *Modelos de séries temporais*):**  
   `Transformer & Zhu et al. (2023) & Mecanismos de atenção em horizontes longos & Exigência de grandes volumes de dados, indisponíveis em DGA aberta \\`
3. **Linha 354 (Corpo da Tabela 2 – `tab:comparacao_trabalhos`, *Trabalhos Relacionados*):**  
   `Zhu et al. (2023) & Transformers em Séries Temporais & Previsão & Atenção em dependências temporais em sistemas de potência & Custo computacional elevado e demanda de grande volume de dados \\`
4. **Linha 369 (Parágrafo da Seção 2.5 – *Trabalhos Relacionados*):**  
   `...explorado por \citeonline{zhu2023}, confirma a utilidade dos mecanismos de autoatenção...` $\rightarrow$ renderiza como **`(?)`** no PDF.

---

### 3.2. O que é Recomendado Fazer? (Duas Opções para sua Escolha)

#### **Opção A (Recomendada — Mais Cirúrgica e Fiel ao Acervo de Artigos do Projeto)**
Substituir as 4 ocorrências de `Zhu et al. (2023)` / `\citeonline{zhu2023}` pela referência real **`Zhou et al. (2021)` / `\citeonline{zhou2021_informer}`** (que já está cadastrada no `referencias.bib` e cujo PDF está na pasta `artigos/`).
* **Por que faz total sentido técnico?** No artigo do *Informer*, Zhou et al. (2021) propõem uma arquitetura *Transformer* voltada a séries temporais longas e validam o modelo justamente criando o famoso conjunto de dados **ETT (*Electricity Transformer Temperature*)**, coletado em transformadores de potência reais ao longo de 2 anos. Ao mesmo tempo, o artigo evidencia que modelos baseados em *Transformer* requerem dezenas de milhares de pontos temporais para convergência, justificando perfeitamente por que a sua pesquisa adotou a rede **LSTM** em vez de um *Transformer*.
* **O que muda no texto (`fundamentacao.tex`):**
  * **Linha 133:** `\citeonline{zhu2023}` $\rightarrow$ `\citeonline{zhou2021_informer}`
  * **Linha 170 (Tabela 1):** `Zhu et al. (2023)` $\rightarrow$ `Zhou et al. (2021)`
  * **Linha 354 (Tabela 2):**  
    `Zhou et al. (2021) & Transformer (Informer) em Séries Temporais & Previsão & Mecanismo de autoatenção eficiente para dependências longas em transformadores & Custo computacional elevado e demanda de grandes volumes de dados \\`
  * **Linha 369 (Texto de Trabalhos Relacionados):**  
    *"Por fim, o uso de arquiteturas Transformer em séries temporais de transformadores, a exemplo do modelo Informer proposto por `\citeonline{zhou2021_informer}`, confirma a utilidade dos mecanismos de autoatenção para sequências longas, mas esbarra na exigência de grandes volumes de dados de treinamento, raramente disponíveis em bases públicas abertas de DGA."*

---

#### **Opção B (Alternativa — Focar a Tabela 2 Exclusivamente em Trabalhos de DGA/Anomalias)**
Caso você prefira que a **Tabela 2** (*Trabalhos Relacionados*) liste apenas artigos estritamente de DGA/diagnóstico de transformadores (e não o artigo de previsão temporal *Informer*):
1. Na **Tabela 1** (Linhas 133 e 170, que lista famílias gerais de algoritmos, assim como já cita `LeCun et al., 1998` para CNN): substituir `\citeonline{zhu2023}` / `Zhu et al. (2023)` por `\citeonline{zhou2021_informer}` / `Zhou et al. (2021)`.
2. Na **Tabela 2** (Linha 354) e no parágrafo da **Linha 369**:
   * **Sub-opção B1:** Substituir a linha de `Zhu et al. (2023)` na Tabela 2 por **`Katser et al. (2022)` (`\citeonline{katser2022}`)** — que é o artigo real da base temporal de DGA usada no seu Track A e que ainda não possui linha dedicada na Tabela 2 — e no parágrafo da Linha 369 mencionar brevemente `Katser et al. (2022)` e `Zhou et al. (2021)` (`\citeonline{zhou2021_informer}`).
   * **Sub-opção B2:** Simplesmente remover a linha de `Zhu et al. (2023)` da Tabela 2 e o parágrafo da Linha 369.

---

## 4. Resumo dos Arquivos que Serão Modificados na Execução

| Arquivo | Ação Planejada |
| :--- | :--- |
| [`tcc-transformer-fault/src/evaluation/fase6_shap_track_a.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-transformer-fault/src/evaluation/fase6_shap_track_a.py) | Ajuste de limites dos eixos Y (`ylim(0, 142)` e `ylim(0, 0.62)`) para separar a linha de soma média das porcentagens nas barras; mudança da legenda para o canto direito superior (`loc='upper right'`); regeração da **Figura 18**. |
| [`tcc-transformer-fault/src/lstm/multi_horizon_eval.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-transformer-fault/src/lstm/multi_horizon_eval.py) | Geração da **Figura 8** utilizando a escala correta de 12 horas por passo (`h * 12`, estendendo todas as linhas até 1.344h / 8 semanas) e eixo superior sincronizado de 0 a 8 semanas. |
| [`tcc-unisinos/fundamentacao.tex`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/TCC_Final/tcc-unisinos/fundamentacao.tex) | Atualização das 4 ocorrências de `zhu2023` / `Zhu et al. (2023)` conforme a opção escolhida (Opção A recomendada: `zhou2021_informer` / `Zhou et al. (2021)`). |

