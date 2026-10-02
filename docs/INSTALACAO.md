# Instalação

## Requisitos

- Python 3.10 ou superior; ambiente validado com Python 3.12.14.
- Aproximadamente alguns gigabytes livres para ambiente, checkpoint e caches.
- Internet na primeira execução para baixar BERT e SST-2.
- CPU, GPU CUDA ou Apple Silicon com MPS. A CPU funciona, mas é mais lenta.

## Ambiente virtual

macOS e Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O `requirements.txt` declara apenas dependências diretas com faixas de versão. Para reproduzir exatamente o ambiente usado na auditoria, as versões centrais observadas foram Python 3.12.14, PyTorch 2.14.0, Transformers 5.17.0, Datasets 5.0.1, NumPy 2.5.3, pandas 3.0.5 e scikit-learn 1.9.1.

## Verificação

```bash
python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

## Seleção de dispositivo

`modeling.get_device()` usa esta prioridade:

1. `mps`, se disponível;
2. `cuda`, se disponível;
3. `cpu` como fallback.

Não há parâmetro de linha de comando para forçar dispositivo. Para isso, altere `get_device()` ou exponha uma nova opção de configuração.

## Downloads e cache

`AutoTokenizer` e `AutoModelForSequenceClassification` resolvem `bert-base-uncased`; `datasets.load_dataset` resolve `stanfordnlp/sst2`. O cache padrão do Hugging Face fica fora do repositório. Downloads interrompidos podem ser repetidos com o mesmo comando.

## Problemas comuns

- **Memória insuficiente:** reduza `batch_size`, `max_seq_length` ou os tamanhos do pool; feche outros processos que usem GPU/MPS.
- **Execução lenta em CPU:** reduza `max_epochs`, pool e conjunto de teste para uma prova rápida.
- **MPS indisponível:** confirme uma versão do PyTorch com suporte MPS e macOS compatível; o fallback para CPU é automático.
- **Erro no download:** confirme rede, espaço em disco e disponibilidade do Hugging Face.
- **Incompatibilidade binária de NumPy/scikit-learn:** recrie o ambiente e reinstale a partir de `requirements.txt` sem misturar pacotes do sistema.
- **Certificados corporativos:** configure o certificado no ambiente; não desative permanentemente verificação TLS.

## Desinstalação local

O código não instala serviços. Para remover apenas o ambiente regenerável, apague `.venv` manualmente. Modelos e datasets ficam no cache global do Hugging Face e não são removidos junto com o projeto.
