# hugface_textclass/data/data_utils.py

import os
import re
import random
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw

import torch
from transformers import AutoTokenizer, AutoModel

########################################
# Basic Text Cleaning
########################################
def basic_cleaning(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data_from_csv(csv_path, text_col="Tweet", label_col="Label", clean_text=False):
    df = pd.read_csv(csv_path)
    if clean_text:
        df[text_col] = df[text_col].apply(basic_cleaning)
    return df[[text_col, label_col]]

########################################
# Optional Data Augmentation
########################################
def maybe_augment_text(df, text_col="Tweet", label_col="Label", 
                       n_aug=2, top_k=5, only_label=None):
    """
    If only_label is set (e.g. 1), only that label is augmented.
    Otherwise, all rows get augmentation attempts.
    """
    aug = naw.ContextualWordEmbsAug(
        model_path="distilbert-base-uncased",
        action="substitute",
        top_k=top_k
    )
    new_texts = []
    new_labels = []

    for i, row in df.iterrows():
        txt = row[text_col]
        lbl = row[label_col]
        # if user wants to augment only certain label
        if only_label is not None and lbl != only_label:
            new_texts.append(txt)
            new_labels.append(lbl)
            continue

        variants = [txt]  # original
        # randomly pick how many variants to create (0..n_aug)
        aug_count = random.randint(0, n_aug)
        if aug_count > 0:
            augmented = aug.augment(txt, n=aug_count)
            for v in augmented:
                variants.append(v)
        for v in variants:
            new_texts.append(v)
            new_labels.append(lbl)

    out_df = pd.DataFrame({text_col: new_texts, label_col: new_labels})
    out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return out_df

########################################
# Creating HF Datasets
########################################
def create_hf_dataset(df, text_col="Tweet", label_col="Label", 
                      val_split=0.1, random_seed=42):
    full_data = Dataset.from_pandas(df)
    if val_split > 0.0 and val_split < 1.0:
        train_dict, val_dict = train_test_split(full_data, 
                                                test_size=val_split,
                                                stratify=full_data[label_col],
                                                random_state=random_seed)
        train_ds = Dataset.from_dict(train_dict)
        val_ds = Dataset.from_dict(val_dict)
    else:
        train_ds = full_data
        val_ds = None
    return train_ds, val_ds

########################################
# Baseline Embedding Pipeline
########################################
def compute_baseline_embeddings(df, text_col="Tweet", label_col="Label", 
                                model_name="distilbert-base-uncased"):
    """
    For the baseline approach, we get a single embedding vector per row
    by passing the text through a frozen Transformer model.
    Returns: embeddings (numpy array) of shape [N, hidden_size], 
             labels (numpy array).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # We'll extract the [CLS] token from AutoModel (not classification head)
    base_model = AutoModel.from_pretrained(model_name)
    base_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    all_embeddings = []
    all_labels = []

    for i, row in df.iterrows():
        txt = row[text_col]
        lbl = row[label_col]
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        with torch.no_grad():
            outputs = base_model(**inputs)
            # For DistilBERT, the last_hidden_state is shape [1, seq_len, hidden_size]
            # The first token is effectively the [CLS] representation:
            cls_emb = outputs.last_hidden_state[:, 0, :]  # shape [1, hidden_size]
            emb = cls_emb.squeeze(0).cpu().numpy()       # shape [hidden_size]
        all_embeddings.append(emb)
        all_labels.append(lbl)
    
    import numpy as np
    embeddings = np.stack(all_embeddings, axis=0)
    labels = np.array(all_labels)
    return embeddings, labels
