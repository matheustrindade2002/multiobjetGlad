# API Python

Esta é uma API de módulos Python, não uma API HTTP. Assinaturas sem anotação completa refletem o código atual.

## `config`

### `ModelConfig`

Dataclass com checkpoint, semente, dimensão de entrada e hiperparâmetros de treino. A instância global é `MODEL_CFG`.

### `DataConfig`

Dataclass com identificação do dataset, nomes de classe e tamanhos dos splits. A instância global é `DATA_CFG`.

### `SelectionConfig`

Dataclass com `k_knn`, piso, margem bootstrap, quantidade de réplicas e orçamento. A instância global é `SEL_CFG`.

O módulo também expõe `BASE_DIR`, `DATA_DIR`, `OUTPUT_DIR` e `PLOTS_DIR` como `pathlib.Path`.

## `data_utils`

### `SplitData(texts, labels)`

Dataclass simples. `texts` é uma lista ou sequência de strings; `labels` é um array NumPy de IDs inteiros.

### `TextClassificationDataset(texts, labels, tokenizer, max_length)`

Implementa `torch.utils.data.Dataset`. `__getitem__` retorna:

```python
{
    "input_ids": Tensor[max_length],
    "attention_mask": Tensor[max_length],
    "labels": Tensor[]
}
```

Cada item é tokenizado sob demanda com padding fixo e truncamento.

### `build_splits()`

Baixa SST-2, embaralha o split de treino e retorna `(labeled, unlabeled, test)` como três `SplitData`. Levanta `ValueError` se não houver exemplos suficientes.

### `make_torch_dataset(split_data, tokenizer)`

Adapta `SplitData` para `TextClassificationDataset` usando `MODEL_CFG.max_seq_length`.

## `modeling`

### `get_device()`

Retorna `torch.device`, priorizando MPS, CUDA e CPU nessa ordem.

### `set_seed(seed)`

Semeia `random`, NumPy, PyTorch CPU e todas as GPUs CUDA disponíveis. Não força algoritmos determinísticos.

### `get_tokenizer()`

Carrega o tokenizer de `MODEL_CFG.model_name`.

### `build_model(device)`

Carrega `AutoModelForSequenceClassification`, configura número de classes e dropouts e move o modelo ao dispositivo.

### `free_model(model)`

Remove a referência local, chama `gc.collect()` e limpa o cache do acelerador disponível. Outras referências externas ainda impedem liberação completa.

### `train(model, tokenizer, train_dataset, device, log_prefix="")`

Divide `train_dataset` em treino/validação interna, otimiza com AdamW, aplica early stopping e restaura o melhor estado. `tokenizer` permanece na assinatura, embora não seja usado diretamente. Retorna `None` e modifica `model`.

Levanta `ValueError` se o dataset não comportar ao menos uma instância de treino e uma de validação.

### `evaluate(model, tokenizer, eval_dataset, device, class_names, log_prefix="")`

Retorna `(metrics, predictions)`. `metrics` contém `accuracy`, `macro_f1` e `per_class_f1`; `predictions` é uma lista de IDs. `tokenizer` e `class_names` não são usados na implementação atual.

### `score_pool(model, tokenizer, pool_dataset, device)`

Retorna um dicionário de tensores CPU:

```python
{
    "pred_class": Tensor[n],
    "confidence": Tensor[n],
    "embeddings": Tensor[n, hidden_size],
}
```

`tokenizer` não é usado diretamente porque o dataset já entrega tokens.

## `selection`

### `get_typicality_and_delta(embeddings, k_knn)`

Calcula tipicidade normalizada e não-centralidade por kNN euclidiano, excluindo o próprio ponto. Retorna dois arrays NumPy de tamanho `n`. Valida pool vazio e `k_knn < 1`.

### `get_robust_expected_confidence(tau, conf, n_bootstraps, rng=None)`

Ajusta regressões isotônicas em réplicas bootstrap. `rng` pode ser um `numpy.random.Generator`. Retorna `(expected_conf, w_tau)`. Valida `n_bootstraps >= 1`.

### `get_pareto_frontier(scores_a, scores_b)`

Retorna os índices NumPy dos pontos não dominados quando ambos os escores são maximizados. Empates exatos permanecem na fronteira.

### `select_pseudo_labels(pred_class, confidence, embeddings, num_classes, sel_cfg, seed)`

Executa seleção por classe. Retorna `(diagnostics_df, moo_selected_idx, naive_selected_idx)`. Os índices referem-se ao pool original. A semente controla bootstrap e K-Means.

## `plotting`

Todas as funções salvam PNG em `PLOTS_DIR`, fecham a figura e retornam o caminho, salvo quando indicado.

- `plot_confidence_vs_typicality(sel_df)`;
- `plot_pareto_frontier(sel_df)`;
- `plot_embedding_selection(sel_df, embeddings)` — PCA 2D;
- `plot_umap_selection(sel_df, embeddings, seed=42)` — UMAP 2D;
- `plot_precision_by_typicality_bucket(sel_df, true_labels)` — retorna também o DataFrame agregado;
- `plot_performance_comparison(metrics_dict)`.

## `pipeline`

### `run_full_pipeline()`

Executa todo o experimento, sobrescreve artefatos de nomes fixos e retorna o dicionário final. Pode propagar erros de rede, memória, configuração, bibliotecas e escrita em disco.

Funções iniciadas por `_` são internas e podem mudar sem compatibilidade: `_dataset_report`, `_write_results_markdown` e `_plot_2d_embedding_selection`.
