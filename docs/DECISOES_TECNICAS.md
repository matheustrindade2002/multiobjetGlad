# Decisões técnicas

## BERT base uncased

Foi escolhido por ser um baseline conhecido, com 768 dimensões ocultas e integração direta ao Transformers. Ele oferece representações contextualizadas, mas é pesado para um experimento tão pequeno e é voltado a inglês, como o SST-2. Alternativas incluem DistilBERT para velocidade e modelos de sentença para geometrias mais adequadas a distância.

## Embedding `[CLS]` da última camada

Evita uma etapa adicional de representação e mantém a seleção ligada ao modelo classificador. A desvantagem é que `[CLS]` não é necessariamente isotrópico nem otimizado para distância euclidiana. *Mean pooling*, normalização ou embeddings contrastivos merecem comparação.

## Densidade kNN em vez de distância a um centróide

Um único centróide presume classe aproximadamente unimodal. A vizinhança local acomoda múltiplos agrupamentos. O custo é maior e o resultado depende de `k`, escala e métrica. O próprio ponto é explicitamente excluído para que `k` conte vizinhos reais.

## Regressão isotônica

Ela modela uma relação crescente sem escolher forma linear ou polinomial. Isso reduz suposições paramétricas, mas impõe monotonicidade e pode criar patamares. Se confiança e tipicidade não forem monotônicas, GAMs, regressão local ou calibração condicionada podem ser mais apropriados.

## Bootstrap como penalização

O desvio entre curvas fornece um sinal simples de instabilidade local. A margem `r - z*w` é conservadora, porém não é um intervalo formal com cobertura garantida. A semente é injetada por `numpy.random.Generator` para repetibilidade isolada.

## Pareto sem escalarização

Não escolher pesos entre qualidade e exploração preserva soluções com compromissos distintos. Em troca, a fronteira pode ser grande e exige uma política posterior de orçamento. Camadas sucessivas mantêm a ordem parcial em vez de introduzir uma soma ponderada.

## K-Means na última camada

K-Means é usado somente quando uma camada Pareto não cabe. Um medóide aproximado — o ponto mais próximo de cada centróide — garante que a saída seja uma instância real. A decisão é pragmática; não equivale a resolver *facility location* e favorece clusters aproximadamente esféricos sob distância euclidiana.

## Orçamento uniforme por classe prevista

A divisão `floor(B/C)` impede que a classe mais frequente ocupe todo o orçamento e facilita comparação. Ela pode superamostrar classes raras ou desperdiçar vagas. Não há redistribuição automática.

## Retreino a partir do checkpoint base

Naive e MOO começam de `bert-base-uncased`, não dos pesos do estágio 0. Isso isola o efeito do conjunto aumentado e evita continuar uma trajetória de otimização específica. As sementes são reiniciadas antes de cada estratégia para tornar inicialização e embaralhamento comparáveis.

## Early stopping com validação interna

Reservar parte do conjunto recebido evita usar o hold-out final para decidir época. Com somente 100 exemplos, porém, a validação é ruidosa e reduz ainda mais o treino efetivo. Validação cruzada ou várias sementes seria metodologicamente mais forte.

## Artefatos experimentais versionados

CSV, JSON e PNG pequenos são incluídos para permitir inspeção sem executar BERT. Pesos, caches e checkpoints são excluídos por tamanho e por serem regeneráveis. Como os nomes são fixos, novos experimentos devem registrar semente, commit e ambiente para evitar ambiguidade.

## Correções feitas na preparação do repositório

- associação dos rótulos verdadeiros aos diagnósticos passou a usar `global_idx`, pois o DataFrame é reordenado por classe prevista;
- kNN passou a excluir a distância do ponto para ele mesmo e a adaptar `k` em classes pequenas;
- bootstrap passou a receber gerador pseudoaleatório semeado;
- Python, NumPy e PyTorch são semeados; os retreinos reiniciam a semente.
- a conversão das saídas do pool passou por um único array NumPy antes de criar tensores, evitando a conversão lenta de uma lista de arrays.

Os artefatos versionados foram regenerados depois dessas mudanças, usando a configuração padrão e MPS.
