# hugface_textclass/training/training_utils.py

import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, EvalPrediction, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

##################################################
# Metrics for HF Trainer
##################################################
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", zero_division=1)
    rec = recall_score(labels, preds, average="binary", zero_division=1)
    f1 = f1_score(labels, preds, average="binary", zero_division=1)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

##################################################
# Baseline Training
##################################################
def baseline_data_split(embeddings, labels, test_size=0.1, batch_size=32):
    """
    Splits the embeddings + labels into train/val sets, 
    returns Dataloaders. 
    """
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=test_size, 
        random_state=42, stratify=labels
    )
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                               torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                             torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_baseline(baseline_model, train_loader, val_loader=None, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model.to(device)
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        baseline_model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = baseline_model(features)   # shape [batch]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f}")

        if val_loader is not None:
            baseline_model.eval()
            preds_list = []
            labs_list = []
            with torch.no_grad():
                for feats, labs in val_loader:
                    feats, labs = feats.to(device), labs.to(device)
                    out = baseline_model(feats)
                    preds = (torch.sigmoid(out) > 0.5).float()
                    preds_list.extend(preds.cpu().numpy())
                    labs_list.extend(labs.cpu().numpy())
            f1 = f1_score(labs_list, preds_list, average="binary", zero_division=1)
            acc = accuracy_score(labs_list, preds_list)
            print(f"  Val Accuracy: {acc:.4f}, Val F1: {f1:.4f}")

    return baseline_model

##################################################
# HF Trainer Pipeline
##################################################
def run_hf_trainer(model, train_ds, val_ds,
                   batch_size=16,
                   lr=2e-5,
                   epochs=3,
                   output_dir="./outputs",
                   text_col="Tweet",
                   label_col="Label",
                   model_name="distilbert-base-uncased"):
    """
    Sets up a standard HF TrainingArguments and Trainer with compute_metrics,
    re-tokenizes the dataset if needed.
    """

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    # if the dataset isn't tokenized yet, we handle that below
    # but in cli.py we might do it anyway

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Hugging Face Trainer completed. Best model in {output_dir}")
