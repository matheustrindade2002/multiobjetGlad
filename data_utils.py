"""
data_utils.py
=============
Baixa e processa o dataset SST-2, criando os splits controlados.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from config import DATA_CFG, MODEL_CFG

@dataclass
class SplitData:
    texts: list
    labels: np.ndarray

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def build_splits():
    """Baixa o SST-2 e divide nas proporções definidas no config."""
    print(f"Carregando dataset {DATA_CFG.dataset_name} via HuggingFace...")
    dataset = load_dataset("stanfordnlp/sst2")

    # Misturar dados para garantir distribuição aleatória
    train_data = dataset["train"].shuffle(seed=MODEL_CFG.seed)

    # Extrair listas
    all_texts = train_data["sentence"]
    all_labels = np.array(train_data["label"])

    n_labeled = DATA_CFG.num_labeled_train
    n_pool = DATA_CFG.num_unlabeled_pool
    n_test = DATA_CFG.num_test

    if len(all_texts) < (n_labeled + n_pool + n_test):
        raise ValueError("Dataset não tem exemplos suficientes para os tamanhos requisitados.")

    # Dividir os splits
    labeled_texts = all_texts[:n_labeled]
    labeled_labels = all_labels[:n_labeled]

    pool_texts = all_texts[n_labeled : n_labeled + n_pool]
    pool_labels = all_labels[n_labeled : n_labeled + n_pool]

    test_texts = all_texts[n_labeled + n_pool : n_labeled + n_pool + n_test]
    test_labels = all_labels[n_labeled + n_pool : n_labeled + n_pool + n_test]

    labeled = SplitData(labeled_texts, labeled_labels)
    unlabeled = SplitData(pool_texts, pool_labels)
    test = SplitData(test_texts, test_labels)

    return labeled, unlabeled, test

def make_torch_dataset(split_data: SplitData, tokenizer):
    return TextClassificationDataset(
        split_data.texts, split_data.labels, tokenizer, MODEL_CFG.max_seq_length
    )