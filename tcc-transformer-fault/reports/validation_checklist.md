# Protocolo de Validação Experimental

Este documento registra os critérios metodológicos adotados no projeto e o histórico de validações concluídas ao longo das fases de desenvolvimento. Funciona como um log técnico do progresso experimental — não como um formulário burocrático, mas como um registro do que foi feito, como foi feito e quais decisões foram tomadas no caminho.

---

## Critérios Metodológicos

### 1. Separação de Dados e Prevenção de Vazamento

- O conjunto de teste nunca é utilizado durante o treinamento, busca de hiperparâmetros ou seleção de arquitetura.
- O modelo LSTM é avaliado exclusivamente nas 900 séries do *split* de teste oficial da base Katser.
- O modelo XGBoost será avaliado em um *split* de teste reservado e separado antes do treinamento (15% da base de classificação).
- A verificação de duplicatas e intersecção treino/teste foi executada antes de qualquer etapa de modelagem.

### 2. Reprodutibilidade e Avaliação Estatística

- Todos os resultados são reportados como **média ± desvio padrão** de múltiplas execuções com *seeds* distintas (mínimo 5), nunca como o resultado de uma única execução.
- A variância entre execuções é reportada. Alta variância indica instabilidade e deve ser discutida.

### 3. Comparação com Baselines

Os resultados do modelo proposto devem ser comparados com:
- Métodos físicos tradicionais: Triângulo de Duval, Método de Rogers.
- Baselines algorítmicos: XGBoost padrão (sem IGA), XGBoost + IGA (sem atributos temporais da LSTM).
- Literatura recente: resultados de Katser et al. (2022) como referência para o contexto do dataset.

### 4. Análise Granular de Erros

- Métricas agregadas (Acurácia, F1-Macro) são sempre acompanhadas da Matriz de Confusão para identificar vieses por classe minoritária.
- Os erros de predição são avaliados quanto à plausibilidade física: confusão entre classes termicamente próximas (T2↔T3) é esperada e aceitável; confusão entre classes distintas (PD↔T3) indica problema metodológico.

### 5. Interpretabilidade (SHAP)

- Os atributos de maior importância identificados pelo SHAP devem ter respaldo físico-químico (ex: razões IEC 60599, concentração de C2H2 para descargas).
- Atributos sem justificativa termoquímica com alta importância são investigados como possível *data leakage* indireto.

---

## Histórico de Validações

| Data | Componente | Status | Resultado / Observações |
|---|---|---|---|
| Ago/2026 | Base de Classificação (DGA compilada) | ✅ Concluído | 349 duplicatas exatas removidas (≈15% da base). Processo documentado em `data/README.md`. |
| Ago/2026 | Base Temporal (Katser et al., 2022) | ✅ Concluído | Zero duplicatas entre treino e teste. Detalhes em `reports/metrics/katser_validation_report.json`. |
| Set/2026 | Treinamento LSTM — Janela Ótima (Fase 3) | ✅ Concluído | Janela ótima **w\* = 50** (600h). R² > 0.999 em escala original (ppm). Média de 5 *seeds* com escalador ajustado apenas no treino. |
| Set/2026 | Avaliação de Múltiplos Horizontes (Fase 3) | ✅ Concluído | Testados 13 horizontes de previsão recursiva (h = 1 a 126 passos de 12h). Critério de aceitação: R² médio ≥ 0.80 entre os 4 gases. **Horizonte máximo aceitável: h = 112 passos (1.344h ≈ 8 semanas), com R² = 0.8388.** No horizonte h = 126 (9 semanas), o R² cai para ≈ 0.78, confirmando o limite. Resultados completos em `reports/metrics/lstm_multi_horizon_results_norm.json` e gráficos em `reports/figures/`. |
