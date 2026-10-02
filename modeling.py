"""
modeling.py
===========
Lida com inicialização, treino, avaliação e inferência do BERT.
Otimizado para MPS (Apple Silicon).
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from config import MODEL_CFG
import gc

def get_device():
    """Detecta Apple Silicon (MPS), CUDA ou cai para CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def set_seed(seed):
    """Semeia as fontes de aleatoriedade usadas diretamente pelo projeto."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_CFG.model_name)

def build_model(device):
    """Instancia o BERT puro, aplicando o dropout configurado."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CFG.model_name,
        num_labels=MODEL_CFG.num_labels,
        hidden_dropout_prob=MODEL_CFG.dropout_prob,
        attention_probs_dropout_prob=MODEL_CFG.dropout_prob
    )
    return model.to(device)

def free_model(model):
    """Libera o modelo da memória RAM/VRAM agressivamente."""
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def train(model, tokenizer, train_dataset, device, log_prefix=""):
    validation_size = max(1, round(len(train_dataset) * MODEL_CFG.validation_fraction))
    training_size = len(train_dataset) - validation_size
    if training_size < 1:
        raise ValueError("O conjunto de treino precisa ter ao menos duas instâncias para early stopping.")

    generator = torch.Generator().manual_seed(MODEL_CFG.seed)
    training_dataset, validation_dataset = random_split(
        train_dataset, [training_size, validation_size], generator=generator
    )
    dataloader = DataLoader(training_dataset, batch_size=MODEL_CFG.batch_size, shuffle=True)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=MODEL_CFG.batch_size * 2
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_CFG.learning_rate)

    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MODEL_CFG.max_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in validation_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                validation_loss += outputs.loss.item()

        training_loss = total_loss / len(dataloader)
        validation_loss /= len(validation_dataloader)
        print(
            f"[{log_prefix}] Época {epoch+1}/{MODEL_CFG.max_epochs} - "
            f"Loss treino: {training_loss:.4f} | Loss validação: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss - MODEL_CFG.early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= MODEL_CFG.early_stopping_patience:
                print(f"[{log_prefix}] Early stopping na época {epoch+1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

def evaluate(model, tokenizer, eval_dataset, device, class_names, log_prefix=""):
    dataloader = DataLoader(eval_dataset, batch_size=MODEL_CFG.batch_size * 2)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class_f1 = f1_score(all_labels, all_preds, average=None)

    print(f"[{log_prefix}] Acurácia: {acc:.4f} | Macro-F1: {macro_f1:.4f}")

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(f) for f in per_class_f1]
    }
    return metrics, all_preds

def score_pool(model, tokenizer, pool_dataset, device):
    """Retorna predições, confianças e embeddings do pool para seleção."""
    dataloader = DataLoader(pool_dataset, batch_size=MODEL_CFG.batch_size * 2)
    model.eval()

    all_preds, all_confs, all_embeddings = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # extraindo output states para pegar o embedding (CLS)
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            # Embedding: último hidden state do token [CLS] (índice 0)
            hidden_states = outputs.hidden_states[-1]
            cls_embeddings = hidden_states[:, 0, :]

            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            all_embeddings.extend(cls_embeddings.cpu().numpy())

    return {
        "pred_class": torch.from_numpy(np.asarray(all_preds)),
        "confidence": torch.from_numpy(np.asarray(all_confs)),
        "embeddings": torch.from_numpy(np.asarray(all_embeddings)),
    }
