"""
config.py
=========
Arquivo central de configurações para o experimento de Pseudo-Rotulagem
Multiobjetivo (MOO). Concentra caminhos de diretório, hiperparâmetros
fixos do modelo, tamanho dos splits (para rodar em hardware doméstico)
e os parâmetros específicos da modelagem da fronteira de Pareto.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# ==========================================
# 1. Diretórios do Projeto
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = BASE_DIR / "plots"

# Garantir que os diretórios de saída existam assim que o config for carregado
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Configurações do Modelo e Treinamento
# ==========================================
@dataclass
class ModelConfig:
    # Arquitetura base do HuggingFace
    model_name: str = "bert-base-uncased"
    seed: int = 42
    num_labels: int = 2

    # max_seq_length mantido em 128 (pode subir até 512, mas 128
    # economiza muita RAM e acelera o treino)
    max_seq_length: int = 128

    # Hiperparâmetros de treinamento
    batch_size: int = 4
    learning_rate: float = 2e-5
    dropout_prob: float = 0.1 # Ajustado via configuração do modelo HuggingFace no modeling.py
    max_epochs: int = 20
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    validation_fraction: float = 0.2

# ==========================================
# 3. Configurações dos Dados (Restrito para RAM)
# ==========================================
@dataclass
class DataConfig:
    # Usando o GLUE SST-2 (Stanford Sentiment Treebank) da literatura
    dataset_name: str = "sst2"
    class_names: List[str] = field(default_factory=lambda: ["Negativo", "Positivo"])

    # "Poucos dados": limitamos severamente o tamanho para rodar rápido no Mac
    num_labeled_train: int = 100    # Estágio 0: simula um cenário de extreme low-resource
    num_unlabeled_pool: int = 1500  # Pool de dados não rotulados para seleção
    num_test: int = 500             # Hold-out puro, NUNCA usado no treino ou seleção

# ==========================================
# 4. Configurações da Seleção Multiobjetivo (MOO)
# ==========================================
@dataclass
class SelectionConfig:
    # Passo 1: Tipicidade kNN
    k_knn: int = 15                 # Quantidade de vizinhos para medir densidade local

    # Passo 4: Robustez (Piso e Incerteza)
    tau_c: float = 0.7              # Piso absoluto: ignora instâncias com confiança menor que 70%
    z_margin: float = 1.0           # Fator Z para margem de erro da regressão isotônica (r_lower = r - z*w)
    n_bootstraps: int = 100          # Quantidade de reamostragens do U_c para estimar a banda w(tau)

    # Passo 6: Facility Location / Tamanho do Conjunto
    # Seleciona as N melhores instâncias que sobreviverem à fronteira de Pareto
    target_selection_size: int = 100


# Instâncias globais importadas pelos outros módulos (pipeline.py, etc.)
MODEL_CFG = ModelConfig()
DATA_CFG = DataConfig()
SEL_CFG = SelectionConfig()
