# hugface_textclass/training/training_utils.py

import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='binary', zero_division=1)
    rec = recall_score(labels, preds, average='binary', zero_division=1)
    f1 = f1_score(labels, preds, average='binary', zero_division=1)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def train_baseline(baseline_model, train_loader, val_loader, epochs=10, lr=1e-3):
    """
    Custom training loop for baseline. 
    This is optional if we want to do baseline outside HF Trainer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model.to(device)
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        baseline_model.train()
        epoch_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = baseline_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")
        
        # Evaluate on val set if provided
        if val_loader:
            baseline_model.eval()
            preds_list = []
            labels_list = []
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = baseline_model(features)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    preds_list.extend(preds.cpu().numpy())
                    labels_list.extend(labels.cpu().numpy())
            # compute F1, etc.
            f1 = f1_score(labels_list, preds_list, average='binary')
            print(f"Validation F1: {f1:.4f}")

    # Return final model
    return baseline_model

