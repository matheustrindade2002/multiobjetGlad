# Limitações

## Validade experimental

- Há uma única execução registrada. Não se pode concluir superioridade estatística do MOO.
- O hold-out vem do split de treino do SST-2, não de um teste externo nem do teste oficial oculto.
- Avaliar três estratégias no mesmo hold-out é válido para comparação, mas ajustar decisões após observar seus resultados causaria sobreajuste experimental.
- O conjunto inicial de 100 exemplos e a validação interna são pequenos; pequenas mudanças de semente podem alterar muito o resultado.
- Os artefatos atuais correspondem ao código corrigido, mas continuam representando somente uma execução em uma combinação de hardware e versões.

## Modelagem

- Probabilidades softmax não são calibradas. `0,7` não garante 70% de acerto.
- Monotonicidade entre confiança e tipicidade é assumida pela regressão isotônica.
- O desvio bootstrap usado em `r_lower` não constitui intervalo de confiança formal.
- Tipicidade é min-max por classe prevista; extremos têm influência forte e escalas não são comparáveis entre classes.
- Erros da classe prevista afetam simultaneamente o agrupamento, a densidade e o balanceamento.
- Distância euclidiana sobre `[CLS]` pode sofrer concentração em alta dimensão e anisotropia.
- K-Means é heurístico e não é *facility location* exato.
- O baseline naive não usa o mesmo piso de confiança do MOO; isso deve ser lembrado ao atribuir causalidade.

## Orçamento e casos extremos

- `target_selection_size // num_classes` descarta o resto da divisão.
- Vagas não usadas por uma classe não são redistribuídas.
- Uma classe sem predições é ignorada.
- Se poucos candidatos superarem o piso, MOO retorna menos exemplos que o alvo.
- Empates exatos permanecem na mesma fronteira e podem acionar K-Means.

## Reprodutibilidade

- Sementes são configuradas, mas kernels MPS/CUDA, operações paralelas e versões de bibliotecas ainda podem ser não determinísticos.
- Faixas em `requirements.txt` facilitam instalação, mas não congelam ambiente bit a bit.
- O download remoto pode fornecer revisões diferentes se o checkpoint/dataset não for fixado por revisão.
- Tempos dependem de hardware, temperatura, cache e rede.

## Engenharia

- Não há CLI, pacote instalável, logging estruturado, checkpoint intermediário ou retomada após falha.
- Configuração global mutável dificulta concorrência e testes de integração.
- Saídas têm nomes fixos e são sobrescritas.
- As dataclasses não validam integralmente tipos e intervalos.
- Algumas assinaturas conservam argumentos atualmente não usados (`tokenizer`, `class_names`).
- A busca de Pareto por pares e o kNN exato não escalam bem para pools grandes.
- Não há testes ponta a ponta automatizados que baixem dados e treinem o modelo.

## Generalização e riscos

- O projeto foi calibrado para sentimento binário em inglês; outras línguas, domínios e múltiplas classes exigem validação.
- Pseudo-rotulagem pode reforçar vieses e erros do modelo inicial.
- Textos reais podem conter dados pessoais ou sensíveis, embora SST-2 seja público.
- Métricas agregadas não capturam custo assimétrico, robustez fora de domínio ou calibração.

## Próximos passos recomendados

1. executar várias sementes e reportar média, desvio-padrão e intervalos;
2. adicionar calibração e curvas de confiabilidade;
3. comparar métricas e representações de embedding;
4. tornar piso e orçamento equivalentes nos baselines quando a pergunta experimental exigir;
5. versionar cada execução com configuração, revisão do dataset e commit;
6. implementar CLI e testes de integração reduzidos;
7. usar kNN aproximado e Pareto eficiente para pools maiores.
