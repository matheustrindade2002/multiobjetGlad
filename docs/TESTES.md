# Testes

## Execução

```bash
python -m unittest discover -s tests -v
python -m compileall -q . -x '(^|/)(\.venv|\.git)/'
```

Os testes são rápidos, não baixam modelos nem datasets e rodam em CPU.

## Cobertura atual

`tests/test_selection.py` verifica:

- exclusão do próprio ponto no kNN e normalização de tipicidade;
- comportamento de uma classe com uma única instância;
- validação de `k_knn` e de quantidade de bootstraps;
- identificação de pontos dominados e não dominados;
- repetibilidade das curvas bootstrap com o mesmo gerador.

`tests/test_plotting.py` verifica que os rótulos verdadeiros são associados por `global_idx`, não pela ordem das linhas após agrupamento por classe. A figura é escrita em diretório temporário.

## O que não está coberto

- download e integridade do SST-2;
- carregamento real do BERT;
- treino, early stopping e equivalência em MPS/CUDA;
- pipeline completo e esquema de todos os artefatos;
- qualidade estatística ou limiares mínimos de desempenho;
- determinismo bit a bit entre plataformas.

## Estratégia recomendada

Uma suíte futura pode separar:

1. **unitários:** matemática, casos extremos e validação;
2. **integração offline:** tokenizer/modelo minúsculo em fixture local;
3. **integração de rede:** download opcional e cacheado;
4. **smoke experimental:** tamanhos e épocas reduzidos;
5. **regressão estatística:** múltiplas sementes e faixas tolerantes, nunca um valor exato isolado.

Testes que treinam BERT devem ser marcados como lentos e não bloquear verificações rápidas em máquinas sem acelerador.
