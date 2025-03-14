# hugface_textclass/cli.py

import argparse
import sys
import os
import torch

from .data.data_utils import load_data_from_csv, create_hf_dataset, maybe_augment_text
from .models.model_utils import BaselineClassifier, get_transformer_classifier, freeze_early_layers
from .training.training_utils import compute_metrics, train_baseline
from transformers import Trainer, TrainingArguments

def parse_args():
    parser = argparse.ArgumentParser(description="hugface_textclass CLI")
    parser.add_argument("--approach", type=str, default="distilbert", 
                        help="Which approach: baseline, distilbert, gpt2, etc.")
    parser.add_argument("--train_file", type=str, required=False, 
                        help="Path to training CSV.")
    parser.add_argument("--text_col", type=str, default="Tweet", 
                        help="Name of text column.")
    parser.add_argument("--label_col", type=str, default="Label", 
                        help="Name of label column.")
    parser.add_argument("--clean_text", action="store_true", 
                        help="If set, apply text cleaning.")
    parser.add_argument("--augment", action="store_true", 
                        help="If set, apply text augmentation.")
    parser.add_argument("--val_split", type=float, default=0.1, 
                        help="Fraction for validation split if no separate val set.")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--freeze_layers", type=int, default=0, help="How many layers to freeze for Transformers.")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save checkpoints.")
    parser.add_argument("--approach_model_name", type=str, default="distilbert-base-uncased", 
                        help="Which Hugging Face model checkpoint to use for approach.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data
    if not args.train_file:
        print("No train_file specified. Exiting.")
        sys.exit(1)
    df = load_data_from_csv(args.train_file, text_col=args.text_col, label_col=args.label_col, clean_text=args.clean_text)

    # Possibly augment
    if args.augment:
        df = maybe_augment_text(df, text_col=args.text_col, label_col=args.label_col, n_aug=2, top_k=5)

    # Convert to HF Datasets
    train_ds, val_ds = create_hf_dataset(df, text_col=args.text_col, label_col=args.label_col, 
                                         val_split=args.val_split, random_seed=42)

    if args.approach.lower() == "baseline":
        print("Running baseline approach with fixed embeddings + MLP classifier.")
        # Here you'd compute embeddings for each row if not done already 
        # or run a minimal training loop. 
        # For brevity, skip or handle it. 
        print("This part would handle baseline embeddings, then train the MLP.")
        # ...
    else:
        print(f"Running Transformers approach: {args.approach}, model {args.approach_model_name}")
        # Use HF Trainer
        model = get_transformer_classifier(args.approach_model_name, num_labels=2)
        if args.freeze_layers > 0:
            model = freeze_early_layers(model, args.freeze_layers)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        def tokenize_function(batch):
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.approach_model_name)
            return tok(batch[args.text_col], truncation=True, padding="max_length")

        # Re-tokenize (the naive approach)
        train_ds = train_ds.map(tokenize_function, batched=True)
        val_ds = val_ds.map(tokenize_function, batched=True)
        train_ds = train_ds.rename_column(args.label_col, "labels") if args.label_col in train_ds.features else train_ds
        val_ds = val_ds.rename_column(args.label_col, "labels") if (val_ds and args.label_col in val_ds.features) else val_ds

        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        if val_ds:
            val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model(args.output_dir)
        print(f"Saved final model to {args.output_dir}")

