# Uso

## Execução ponta a ponta

Com o ambiente ativado, na raiz do projeto:

```bash
python -u -c "import pipeline; pipeline.run_full_pipeline()"
```

Não há script executável sob `if __name__ == "__main__"`; a função deve ser importada. O retorno é um dicionário equivalente ao conteúdo salvo em `outputs/results.json`.

## Etapas observáveis

1. detecção de dispositivo;
2. download e criação dos splits;
3. treino do modelo somente rotulado;
4. avaliação baseline no hold-out;
5. pontuação do pool;
6. seleção naive e MOO;
7. retreino naive do checkpoint base;
8. retreino MOO do checkpoint base;
9. análise offline, gráficos e relatórios.

O console mostra perdas de treino/validação por época, early stopping, métricas e tempo final. Interromper o processo pode deixar alguns artefatos antigos no disco, pois não existe diretório transacional por execução.

## Uso programático

```python
from config import DATA_CFG, MODEL_CFG, SEL_CFG
from pipeline import run_full_pipeline

MODEL_CFG.seed = 7
DATA_CFG.num_unlabeled_pool = 500
SEL_CFG.target_selection_size = 50

resultado = run_full_pipeline()
print(resultado["test_metrics"])
```

Faça as alterações antes da chamada. Como as configurações são objetos globais, evite executar experimentos concorrentes no mesmo processo.

## Saídas

| Arquivo | Campos ou conteúdo principal |
|---|---|
| `data/dataset_report.json` | tamanhos, contagem de classes e aviso sobre rótulos do pool |
| `data/selection_diagnostics.csv` | índice global, classe prevista, confiança, tipicidade, delta, expectativa, incerteza, `r_lower` e flags |
| `data/precision_by_bucket.csv` | método, quartil, precisão e quantidade |
| `outputs/results.json` | dispositivo, configuração efetiva, métricas, contagens e tempo |
| `outputs/RESULTS.md` | versão legível das principais métricas |
| `plots/confidence_vs_typicality.png` | confiança, tipicidade, curva isotônica e banda de um desvio |
| `plots/pareto_frontier.png` | objetivos e pontos MOO escolhidos |
| `plots/embedding_pca_selected.png` | comparação espacial linear |
| `plots/embedding_umap_selected.png` | comparação espacial não linear |
| `plots/precision_by_typicality_bucket.png` | precisão offline por quartil |
| `plots/performance_comparison.png` | acurácia e Macro-F1 dos três modelos |

## Leitura de `selection_diagnostics.csv`

- `global_idx`: posição original no pool; chave para alinhar textos e rótulos.
- `class`: classe prevista pelo modelo inicial.
- `confidence`: maior probabilidade softmax.
- `typicality`: densidade kNN normalizada por classe prevista.
- `delta`: `1 - typicality`.
- `expected_conf`: média das previsões isotônicas bootstrap.
- `uncertainty_w`: desvio-padrão entre réplicas.
- `r_lower`: excedente menos penalização de incerteza.
- `selected_moo` e `selected_naive`: inclusão por estratégia.

## Interpretação correta

- A precisão dos pseudo-rótulos mede correção dos itens escolhidos; não mede diversidade nem ganho final.
- Macro-F1 é preferível à acurácia quando o desempenho equilibrado entre classes importa.
- PCA e UMAP distorcem informação; servem para diagnóstico, não para demonstrar cobertura formal.
- Uma execução pode favorecer um método por acaso. Compare várias sementes.
- Não use o hold-out para escolher `tau_c`, `k_knn`, orçamento ou época.

## Execução rápida de desenvolvimento

Para validar o encadeamento sem um experimento completo, reduza os tamanhos conforme [Configuração](CONFIGURACAO.md#exemplo-de-experimento-reduzido). Garanta que cada classe prevista tenha exemplos suficientes; classes muito pequenas ativam o `k` efetivo reduzido e podem não preencher o orçamento.

## Regeração de artefatos

O pipeline sobrescreve arquivos de nomes fixos. Antes de comparar execuções, copie os artefatos para um diretório identificado pela semente ou implemente versionamento por execução. Não edite manualmente `results.json` e os CSVs se eles forem usados como evidência experimental.
