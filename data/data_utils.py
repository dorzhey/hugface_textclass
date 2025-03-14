# hugface_textclass/data/data_utils.py

import os
import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import nlpaug.augmenter.word as naw

def basic_cleaning(text):
    """
    Removes URLs and non-ASCII, lowercases, punctuation, etc.
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data_from_csv(csv_path, text_col="Tweet", label_col="Label", clean_text=False):
    """
    Loads data from a CSV file, optionally applying text cleaning.
    Returns a Pandas DataFrame with at least two columns: text_col and label_col.
    """
    df = pd.read_csv(csv_path)
    if clean_text:
        df[text_col] = df[text_col].apply(basic_cleaning)
    return df[[text_col, label_col]]

def maybe_augment_text(df, text_col="Tweet", label_col="Label", 
                       n_aug=2, top_k=5, only_label=None):
    """
    Example augmentation using nlpaug (ContextualWordEmbsAug).
    If only_label is given (e.g. 1 for minority class), only that class is augmented.
    """
    aug = naw.ContextualWordEmbsAug(model_path="distilbert-base-uncased",
                                    action="substitute",
                                    top_k=top_k)
    # Build new data
    augmented_texts = []
    augmented_labels = []
    for idx, row in df.iterrows():
        text = row[text_col]
        label = row[label_col]
        if only_label is not None and label != only_label:
            # no augmentation for other labels
            augmented_texts.append(text)
            augmented_labels.append(label)
            continue
        
        # generate augmented versions
        new_texts = [text]  # original
        n = random.randint(0, n_aug)
        if n > 0:
            # e.g. generate n new variants
            new_texts.extend(aug.augment(text, n=n))

        for txt in new_texts:
            augmented_texts.append(txt)
            augmented_labels.append(label)
    
    out_df = pd.DataFrame({text_col: augmented_texts, label_col: augmented_labels})
    out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return out_df

def create_hf_dataset(df, text_col="Tweet", label_col="Label", 
                      val_split=0.1, random_seed=42):
    """
    Converts a Pandas DataFrame into a HF dataset, optionally splitting for validation.
    Returns train_dataset, val_dataset.
    """
    full_data = Dataset.from_pandas(df)
    if val_split > 0.0:
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

