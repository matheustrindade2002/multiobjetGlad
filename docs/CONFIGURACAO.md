# Configuração

Todas as opções atuais são atributos de dataclasses em `config.py`. Não há CLI, arquivo YAML nem leitura de `.env`. Alterações valem no próximo processo Python.

## `ModelConfig`

| Parâmetro | Padrão | Domínio prático | Efeito |
|---|---:|---|---|
| `model_name` | `bert-base-uncased` | ID compatível com Transformers | Define tokenizer, arquitetura e checkpoint. Outro modelo pode mudar a dimensão e o custo. |
| `seed` | `42` | inteiro | Semeia Python, NumPy e PyTorch; também controla split, K-Means e UMAP. |
| `num_labels` | `2` | inteiro >= 2 | Número de saídas do classificador. Deve coincidir com dados e `class_names`. |
| `max_seq_length` | `128` | 1 até o limite do modelo | Mais contexto aumenta memória e tempo aproximadamente com custo quadrático da atenção. |
| `batch_size` | `4` | inteiro >= 1 | Maior lote melhora vazão, mas eleva RAM/VRAM. Avaliação usa o dobro. |
| `learning_rate` | `2e-5` | positivo | Passo do AdamW. Muito alto pode desestabilizar; muito baixo pode convergir lentamente. |
| `dropout_prob` | `0.1` | `[0,1)` | Aplicado ao dropout oculto e de atenção do BERT. |
| `max_epochs` | `20` | inteiro >= 1 | Teto; early stopping normalmente encerra antes. |
| `early_stopping_patience` | `3` | inteiro >= 1 | Épocas consecutivas sem melhora mínima aceitas. |
| `early_stopping_min_delta` | `1e-4` | não negativo | Redução mínima da loss para reiniciar a paciência. |
| `validation_fraction` | `0.2` | `(0,1)` | Fração de cada conjunto de treino reservada internamente. |

O otimizador usa os padrões de `torch.optim.AdamW` para parâmetros não expostos, como `betas`, `eps` e `weight_decay`.

## `DataConfig`

| Parâmetro | Padrão | Efeito |
|---|---:|---|
| `dataset_name` | `sst2` | Nome exibido nos relatórios. O carregamento atual usa explicitamente `stanfordnlp/sst2`. |
| `class_names` | `Negativo`, `Positivo` | Rótulos legíveis, na ordem dos IDs. |
| `num_labeled_train` | `100` | Exemplos iniciais com rótulo. |
| `num_unlabeled_pool` | `1500` | Exemplos tratados como não rotulados durante a seleção. |
| `num_test` | `500` | Hold-out interno usado somente em avaliação. |

É obrigatório que a soma dos três tamanhos não exceda o split de treino disponível. Os segmentos são não sobrepostos após `shuffle(seed)`.

## `SelectionConfig`

| Parâmetro | Padrão | Domínio | Efeito e compromisso |
|---|---:|---|---|
| `k_knn` | `15` | inteiro >= 1 | Maior `k` suaviza a densidade e perde detalhes locais; menor `k` é mais sensível a ruído. O código limita `k` a `n_classe-1`. |
| `tau_c` | `0.7` | normalmente `[0,1]` | Piso de confiança do MOO. Alto reduz ruído e candidatos; baixo amplia exploração. Não se confunde com tipicidade `tau`. |
| `z_margin` | `1.0` | normalmente >= 0 | Multiplicador da incerteza bootstrap. Alto penaliza regiões instáveis. |
| `n_bootstraps` | `100` | inteiro >= 1 | Mais réplicas estabilizam `w`, com custo aproximadamente linear. |
| `target_selection_size` | `100` | inteiro >= 0 | Orçamento global pretendido; usa divisão inteira uniforme por classe. |

O nome `tau_c` representa piso de confiança, enquanto `tau(x)` representa tipicidade. Apesar da notação próxima, são grandezas diferentes.

## Caminhos

`BASE_DIR` é o diretório de `config.py`. Na importação, o módulo cria automaticamente:

- `DATA_DIR = BASE_DIR / "data"`;
- `OUTPUT_DIR = BASE_DIR / "outputs"`;
- `PLOTS_DIR = BASE_DIR / "plots"`.

## Exemplo de experimento reduzido

Edite temporariamente os padrões em `config.py`:

```python
MODEL_CFG.max_epochs = 2
DATA_CFG.num_labeled_train = 40
DATA_CFG.num_unlabeled_pool = 200
DATA_CFG.num_test = 100
SEL_CFG.n_bootstraps = 20
SEL_CFG.target_selection_size = 20
```

Essas atribuições também podem ser feitas antes de chamar o pipeline no mesmo processo. Elas não persistem em disco e devem ocorrer antes de `run_full_pipeline()`.

## Validação ausente

As dataclasses não validam todos os domínios ao serem criadas. Algumas combinações inválidas falham somente dentro de PyTorch ou scikit-learn. Uma evolução recomendada é validar configurações na entrada e registrar um arquivo imutável por execução.
