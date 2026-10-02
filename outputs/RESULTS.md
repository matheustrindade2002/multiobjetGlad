# Resultados da execução

- Dispositivo: `mps`
- Tempo total: 239.1s
- Pseudo-rótulos selecionados: MOO=100, naive=100
- Precisão geral dos pseudo-rótulos (análise offline, vs. rótulo verdadeiro): MOO=0.680, naive=0.910

## Desempenho no hold-out de teste (nunca tocado por pseudo-rotulagem)

| Modelo | Acurácia | Macro-F1 |
|---|---|---|
| Baseline (só rotulado) | 0.6380 | 0.6147 |
| Self-training (naive top-confidence) | 0.7320 | 0.7319 |
| Self-training (MOO: excedente de confiança x não-centralidade) | 0.7140 | 0.7051 |

## Precisão dos pseudo-rótulos por quartil de tipicidade (Q1=periférico ... Q4=central)

| Método | Bucket | n | Precisão |
|---|---|---|---|
| MOO | Q1 (Periférico) | 74 | 0.635 |
| MOO | Q2 | 23 | 0.826 |
| MOO | Q3 | 3 | 0.667 |
| MOO | Q4 (Central) | 0 | n/a |
| Naive | Q1 (Periférico) | 0 | n/a |
| Naive | Q2 | 0 | n/a |
| Naive | Q3 | 3 | 1.000 |
| Naive | Q4 (Central) | 97 | 0.907 |

## Gráficos gerados

- `plots/confidence_vs_typicality.png` -- Confiança vs. tipicidade, por classe, com curva isotônica esperada
- `plots/pareto_frontier.png` -- Fronteira de Pareto (excedente robusto de confiança x não-centralidade)
- `plots/embedding_pca_selected.png` -- Projeção PCA do pool: selecionado (MOO/naive) vs. não selecionado
- `plots/embedding_umap_selected.png` -- Projeção UMAP 2D do pool: selecionado (MOO/naive) vs. não selecionado
- `plots/precision_by_typicality_bucket.png` -- Precisão dos pseudo-rótulos por quartil de tipicidade
- `plots/performance_comparison.png` -- Acurácia/Macro-F1: baseline vs. naive vs. MOO