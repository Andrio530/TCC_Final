# 🔬 Parecer de Auditoria — TCC: Sistema Híbrido LSTM + IGA-XGBoost para Diagnóstico de Falhas em Transformadores

**Auditor:** Cientista de Dados Sênior / Avaliador de Banca  
**Data:** 05/10/2026  
**Repositório auditado:** `tcc-transformer-fault`  
**Escopo:** Auditoria completa de código, metodologia e resultados

---

## 🚨 CRÍTICO — Falhas que comprometem a confiabilidade dos resultados

### CRÍTICO 1: Variância ZERO no multi-seed do C2 (Track A) e C4/C3 (Track B) — Possível bug ou determinismo artificial

Os resultados do experimento **C2_default (Track A)** mostram variância exatamente zero entre as 5 seeds:

```json
"f1_macro_std": 1.1102230246251565e-16,    // = ε de máquina → ZERO real
"accuracy_std": 0.0,
"f1_por_seed": [0.8951, 0.8951, 0.8951, 0.8951, 0.8951]  // idênticos!
```

O mesmo ocorre no **C4_default (Track B)**:
```json
"f1_macro_std": 0.0,
"f1_por_seed": [0.7782, 0.7782, 0.7782, 0.7782, 0.7782]  // idênticos!
```

> [!CAUTION]
> **XGBoost com `tree_method="hist"` e `subsample=1.0`, `colsample_bytree=1.0` é determinístico** — a seed não tem efeito quando não há amostragem estocástica. Esses resultados "multi-seed" são enganosos: parecem 5 avaliações independentes, mas são matematicamente idênticas. **A banca vai identificar imediatamente que o desvio-padrão reportado é artificial e não valida a robustez do modelo.**
>
> **Impacto:** Não invalida os resultados em si, mas invalida a alegação de "avaliação com múltiplas seeds" para os experimentos default. Na defesa, isso será questionado como falta de rigor estatístico.
>
> **Correção:** No texto do TCC, declarar explicitamente que para `subsample=1.0` e `colsample_bytree=1.0`, o XGBoost é determinístico, portanto a avaliação multi-seed não se aplica. Alternativamente, introduzir aleatoriedade real via `subsample < 1.0` nos experimentos default.

### CRÍTICO 2: SMOTE aplicado no CustomSMOTE antes do CV no IGA — Potencial data leakage no fitness

No arquivo [`iga.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/tcc-transformer-fault/src/iga_xgboost/iga.py#L150-L156), o pipeline do IGA é:

```python
pipeline = ImbPipeline([
    ("smote", sampler),     # SMOTE dentro do pipeline
    ("select", selector),
    ("clf", clf),
])
```

Isso está **correto** para o fitness via cross-validation, pois o `ImbPipeline` do `imblearn` aplica o SMOTE dentro de cada fold (somente no treino do fold). ✅

**PORÉM**, no arquivo [`train_classification.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/tcc-transformer-fault/src/iga_xgboost/train_classification.py#L297-L301):

```python
# Linha 298 — SMOTE aplicado em TODO o X_train antes do treino final
X_train_sm, y_train_sm = custom_smote.fit_resample(X_train, y_train)

modelo_iga = _treinar_xgboost(
    X_train_sm[:, features_idx_iga], y_train_sm, hparams_iga, RANDOM_SEED,
)
```

E na avaliação multi-seed (linhas 308-309):
```python
multiseed_c3 = _avaliar_multi_seed(X_train_sm, y_train_sm, X_test, y_test,
                                    hparams_iga, features_idx_iga, classes)
```

**O SMOTE é aplicado em todo o X_train ANTES de passar para `_avaliar_multi_seed`**, que simplesmente treina e avalia no teste. Isso em si **não é data leakage** (o SMOTE é aplicado apenas no treino, nunca no val/teste). ✅

> [!IMPORTANT]
> Revisado: **não há data leakage no SMOTE** — ele é aplicado apenas nos dados de treino. Porém, o fato de aplicar SMOTE sobre todo o treino e depois não re-aplicar por seed gera dados sintéticos idênticos para todas as 5 seeds, reduzindo a validade da análise multi-seed. É um **problema menor**, não crítico.

### CRÍTICO 3: CustomSMOTE reconstrói features de forma fisicamente incorreta

No [`train_classification.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/tcc-transformer-fault/src/iga_xgboost/train_classification.py#L89-L135), o `CustomSMOTE`:

1. Aplica SMOTE apenas nas 5 primeiras colunas (gases normalizados)
2. **Desfaz a normalização** (`inverse_transform` + `expm1`)
3. **Recalcula as 14 features derivadas** a partir dos gases desnormalizados
4. **Refaz a normalização** com o pipeline original

Esse procedimento é **conceitualmente correto** — mantém a consistência física entre gases e razões. ✅

**PORÉM**, há um problema na implementação. Na linha 95-99:

```python
X_dummy = np.zeros((X_res.shape[0], 19))
X_dummy[:, :5] = X_res     # ← X_res ainda está em escala NORMALIZADA (saída do SMOTE)

minmax = self.scaler.named_steps["scaler"]
X_raw = minmax.inverse_transform(X_dummy)
X_raw = np.expm1(X_raw)
```

> [!WARNING]
> O `inverse_transform` do MinMaxScaler é aplicado a um vetor de 19 features onde **as 14 últimas são zeros**. O `inverse_transform` irá desnormalizar esses zeros usando `min_` e `scale_` de features diferentes dos gases brutos, produzindo **valores artificiais não-zero** nas colunas 5-18. Esses valores são imediatamente sobrescritos nas linhas 110-131, então **o resultado final está correto**. Mas é código frágil e confuso — se alguém alterar a ordem das features, o bug se manifestará silenciosamente.

---

## ⚠️ ALERTA — Problemas metodológicos que a banca vai questionar

### ALERTA 1: Normalização (log1p + MinMaxScaler) fitada no treino sem SMOTE — Correto, mas justifique na defesa

Em [`build_classification_features.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/tcc-transformer-fault/src/feature_engineering/build_classification_features.py#L256-L263), o pipeline de normalização é fitado no treino **sem SMOTE**:

```python
pipeline.fit(splits_raw["train"][colunas_features])
```

Isso é **correto** e defensável: os parâmetros de normalização devem refletir a distribuição real, não a distribuição inflada pelo SMOTE. ✅ **Porém, a banca vai perguntar — tenha a justificativa na ponta da língua.**

### ALERTA 2: Split estratificado ALEATÓRIO no Katser — Justificável, mas requer argumentação sólida

O split do Katser (séries temporais) usa `train_test_split` com `stratify` — um split **aleatório**, não temporal. A justificativa está no docstring de [`preprocess_katser.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/tcc-transformer-fault/src/preprocessing/preprocess_katser.py#L10-L17):

> "A ordem de ID NÃO é usada como proxy cronológico entre séries, pois cada série representa um equipamento diferente."

> [!IMPORTANT]
> Essa justificativa é **aceitável** se cada série é um equipamento independente (cross-sectional). A cronologia walk-forward é respeitada **dentro** de cada série pelo windowing. Contudo, a banca pode questionar se não há dependência temporal entre equipamentos (ex: mudança de política de manutenção, lote de fabricação). **Prepare contra-argumento:**
> - O Katser é um benchmark público com séries de NPPs distintas
> - TimeSeriesSplit entre séries não é aplicável sem evidência de ordering temporal entre equipamentos

### ALERTA 3: A LSTM não tem horizonte de previsão no Grupo 3 — Inconsistência entre SMOTE e LSTM

No Track A, o Grupo 3 usa previsões LSTM com horizontes h=14 e h=28 como features. Essas previsões são **recursivas** (a LSTM é treinada one-step-ahead e iterada h vezes). 

> [!WARNING]
> O scaler da LSTM é **reconstruído** em `_reconstruir_scaler` (linha 131 de [`build_katser_features.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/tcc-transformer-fault/src/feature_engineering/build_katser_features.py#L131-L135)):
> ```python
> def _reconstruir_scaler(series_treino_raw: np.ndarray) -> ScalerPerGas:
>     scaler = ScalerPerGas().fit(series_treino_raw)
> ```
> Isso refaz o fit do scaler nos dados de treino brutos — o que é correto e reprodutível. ✅ Mas o fato de re-fitar em vez de carregar o scaler original salvo cria risco de divergência caso os dados mudem.

### ALERTA 4: Acurácia reportada junto com F1-macro — OK, mas contextualize

Em ambos os tracks, tanto a acurácia quanto o F1-macro são reportados. O Katser (Track A) é altamente desbalanceado:
- Normal: 81% (731/900 no teste)
- PD: 4.2%, D1: 5.4%, T1: 9.1%

A acurácia do Track A (95.2%) é inflada pela classe majoritária. **A banca vai questionar se o modelo realmente prevê falhas ou apenas chuta "Normal".**

**Análise da matriz de confusão do C1_IGA (Track A, teste):**
- Normal: 709/731 corretos (97.0%) ✅
- PD: 34/38 corretos (89.5%) ✅
- D1: 47/49 corretos (95.9%) ✅
- **T1: 67/82 corretos (81.7%)** ← pior classe, 15 erros (8 confundidos com Normal)

> [!NOTE]
> O modelo **não está apenas chutando a classe majoritária** — os recalls das classes minoritárias (PD=89.5%, D1=95.9%) são altos. T1 é o ponto fraco (recall 81.7%), o que é esperado dado o número limitado de amostras de treino. **Resposta à banca: F1-macro de 0.89 comprova que o modelo discrimina efetivamente todas as 4 classes.**

### ALERTA 5: Track B (7 classes) — Resultados significativamente piores, T1 com F1=0.67

O Track B (base chinesa, 7 classes IEC) mostra F1-macro de **0.787** no teste — muito inferior ao Track A (0.889). Os pontos fracos:

| Classe | Precision | Recall | F1    | Support |
|--------|-----------|--------|-------|---------|
| T1     | 0.538     | 0.875  | 0.667 | 16      |
| T2     | 0.615     | 0.500  | 0.552 | 16      |
| D2     | 0.763     | 0.879  | 0.817 | 33      |

> [!WARNING]
> - **T1 tem precision de 0.538** — o modelo alerta falso positivo para T1 em 46% dos casos. Isso é problemático em produção (falsas paradas de manutenção).
> - **T2 tem recall de 0.500** — o modelo perde metade das falhas de sobreaquecimento de média temperatura. Isso é **perigoso** em um sistema de confiabilidade.
> - Esses resultados devem ser apresentados com **transparência total** no TCC. Não omita as classes fracas.

### ALERTA 6: Diferença grande entre fitness CV do IGA e métricas de teste

- **Track B C3_IGA:** fitness CV = 0.791, F1 teste = 0.787 → **Gap de ~0.004** ✅ Muito consistente.
- **Track A C1_IGA:** fitness CV = 0.883, F1 teste = 0.889 → **F1 teste SUPERIOR ao CV!**

> [!NOTE]
> O F1 de teste ser marginalmente superior ao CV não é alarme grave — pode ser variância natural. Mas um avaliador rigoroso pode questionar se há leak sutil. A explicação mais provável: o teste oficial do Katser (900 séries) tem uma distribuição ligeiramente mais fácil que a partição interna de 5-fold.

---

## 💡 SUGESTÃO — Melhorias e dicas para a defesa

### SUGESTÃO 1: Torne o multi-seed do default realmente informativo

Para os experimentos C2/C4 (default), como `subsample=1.0` torna o XGBoost determinístico, as opções são:
1. **Remover** a coluna de σ da tabela para esses experimentos e declarar "determinístico"
2. Ou usar `subsample=0.9` no default para introduzir variabilidade real

### SUGESTÃO 2: Adicione intervalo de confiança ao F1 do IGA

O IGA (Track A) tem σ = 0.0011 entre seeds — isso é excelente. Reporte como **F1 = 0.891 ± 0.001** e calcule o intervalo de confiança 95% (com t de Student, n=5): ≈ 0.891 ± 0.0014. Isso impressiona qualquer banca.

### SUGESTÃO 3: Discuta a confusão D1 ↔ D2 no Track B

A matriz de confusão do Track B mostra 6 amostras de D1 confundidas com D2 e 4 de D2 confundidas com D1. Isso é **fisicamente esperado** — D1 (descarga baixa energia) e D2 (descarga alta energia) produzem perfis de gases muito similares, diferenciados principalmente pela intensidade. Cite IEC 60599 e a zona DT do Duval como evidência teórica.

### SUGESTÃO 4: O código está excepcionalmente bem documentado

Os docstrings, comentários e a estrutura de módulos são de qualidade acima da média para um TCC. Os arquivos `config.py` e os relatórios JSON intermediários demonstram rastreabilidade científica. Destaque isso na defesa como evidência de reprodutibilidade.

### SUGESTÃO 5: Mencione o ScalerPerGas como contribuição original

O `ScalerPerGas` fitado exclusivamente no treino, com `inverse_transform` para reportar MAE/RMSE em escala original (ppm), é uma prática de engenharia de ML correta e muitas vezes negligenciada em TCCs. Argumente que seus erros reportados são em unidade física interpretável.

### SUGESTÃO 6: Salve o scaler da LSTM em vez de reconstruí-lo

Em [`build_katser_features.py`](file:///c:/Users/andriogabriel.eppin/.gemini/antigravity-ide/scratch/tcc-transformer-fault/src/feature_engineering/build_katser_features.py#L131-L135), em vez de `_reconstruir_scaler()`, salve o scaler com `pickle` na Fase 3 e carregue-o na Fase 4. Isso elimina riscos de divergência se os dados forem modificados.

### SUGESTÃO 7: Adicione ablação do Grupo 3 (previsões LSTM) no Track A

Para provar que o Grupo 3 (previsões LSTM) agrega valor, compare:
- C1_IGA (todas as features) vs. C1_IGA sem features `lstm_h*_*`
- Se o IGA já desativou automaticamente features LSTM no melhor cromossomo, esse dado já existe implicitamente.

Olhando o resultado: o IGA selecionou 4 features LSTM (`lstm_h14_H2`, `lstm_h28_H2`, `lstm_h28_C2H4`, `lstm_h28_C2H2`) das 8 disponíveis — isto é, **o IGA validou que as previsões LSTM são úteis** para discriminação. Mencione isso na defesa.

---

## 📊 PARECER DOS RESULTADOS — Análise direta e sincera

### Veredito: ✅ RESULTADOS CONFIÁVEIS E CONSISTENTES, com ressalvas menores

#### Track A (Katser, 4 classes) — **SÓLIDO**

| Aspecto | Avaliação |
|---------|-----------|
| Data leakage | ❌ Não detectado |
| Normalização | ✅ Fitada corretamente no treino |
| SMOTE | ✅ Apenas no treino (Track A usa `sample_weight`, ainda melhor) |
| Split | ✅ Justificável (cross-sectional, sem dependência temporal entre séries) |
| Métricas | ✅ F1-macro como métrica principal; recall > 80% em todas as classes |
| Overfitting | ❌ Gap CV→teste < 1% |
| Estabilidade | ✅ σ = 0.001 entre seeds (IGA) |
| **F1-macro final** | **0.891 ± 0.001** — **Excelente para 4 classes desbalanceadas** |

#### Track B (China, 7 classes) — **ACEITÁVEL, com limitações documentáveis**

| Aspecto | Avaliação |
|---------|-----------|
| Data leakage | ❌ Não detectado |
| Normalização | ✅ Fitada corretamente no treino |
| SMOTE | ⚠️ Aplicado corretamente, mas CustomSMOTE tem implementação frágil |
| Split | ✅ Estratificado 70/15/15 |
| Métricas | ⚠️ T1 e T2 com F1 < 0.70 — classes com 16 amostras cada no teste |
| Overfitting | ❌ Gap CV→teste < 1% |
| Estabilidade | ⚠️ σ = 0.012 entre seeds (IGA), multi-seed default é falso (variância zero) |
| **F1-macro final** | **0.774 ± 0.012** — **Aceitável para 7 classes, mas não excepcional** |

#### LSTM (Previsão de gases) — **CONFIÁVEL**

| Aspecto | Avaliação |
|---------|-----------|
| Windowing | ✅ Walk-forward dentro de cada série, sem leakage |
| Scaler | ✅ Fitado apenas no treino |
| Avaliação | ✅ Multi-seed com 5 runs, reporta média ± σ |
| Multi-horizonte | ✅ Degradação coerente com R² threshold de 0.80 |

### Resumo para a defesa

1. **O trabalho NÃO contém data leakage.** A normalização e o SMOTE são aplicados corretamente após o split.
2. **As métricas corretas (F1-macro) são usadas como principal.** A acurácia é reportada como métrica secundária, o que é adequado.
3. **Os resultados NÃO são "bons demais para ser verdade."** O Track A (0.89) é bom mas não perfeito; o Track B (0.77) mostra limitações reais nas classes minoritárias.
4. **O maior risco na defesa** é a banca questionar a variância zero dos experimentos default e as classes T1/T2 fracas no Track B. Prepare as respostas conforme as sugestões acima.
5. **O código é de qualidade profissional** para um TCC — modular, documentado, reprodutível.

> [!TIP]
> **Nota final: 8.5/10.** O trabalho está pronto para defesa com pequenos ajustes de apresentação. Os resultados são honestos, não há manipulação ou vazamento, e a metodologia é fundamentalmente sólida. As limitações do Track B (7 classes com amostras escassas em T1/T2) são inerentes ao dataset e devem ser apresentadas como tal, não como falha do método.
