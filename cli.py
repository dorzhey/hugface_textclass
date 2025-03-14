# hugface_textclass/cli.py

import argparse
import sys
import os
import random

import torch
from .data.data_utils import (load_data_from_csv, 
                              create_hf_dataset, 
                              maybe_augment_text, 
                              compute_baseline_embeddings)
from .models.model_utils import (BaselineClassifier, 
                                 get_transformer_classifier, 
                                 freeze_early_layers)
from .training.training_utils import (compute_metrics, 
                                      train_baseline, 
                                      run_hf_trainer)
from transformers import Trainer, TrainingArguments

def parse_args():
    parser = argparse.ArgumentParser(description="hugface_textclass CLI")

    parser.add_argument("--approach", type=str, default="distilbert", 
                        help="Which approach: baseline, distilbert, or any HF model name.")
    parser.add_argument("--train_file", type=str, required=True, 
                        help="Path to training CSV.")
    parser.add_argument("--text_col", type=str, default="Tweet", 
                        help="Name of text column.")
    parser.add_argument("--label_col", type=str, default="Label", 
                        help="Name of label column.")
    parser.add_argument("--clean_text", action="store_true", 
                        help="Apply text cleaning if set.")
    parser.add_argument("--augment", action="store_true", 
                        help="Apply data augmentation if set.")
    parser.add_argument("--val_split", type=float, default=0.1, 
                        help="Fraction for validation split [0,1].")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate.")
    parser.add_argument("--freeze_layers", type=int, default=0, 
                        help="How many layers to freeze if fine-tuning a Transformer.")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save checkpoints.")
    parser.add_argument("--approach_model_name", type=str, default="distilbert-base-uncased", 
                        help="Which HF model to use if approach != baseline.")
    # Additional arguments can be appended as needed (logging steps, etc.)

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load raw data
    df = load_data_from_csv(
        args.train_file, 
        text_col=args.text_col, 
        label_col=args.label_col, 
        clean_text=args.clean_text
    )

    # 2. Possibly augment
    if args.augment:
        df = maybe_augment_text(df, 
                                text_col=args.text_col, 
                                label_col=args.label_col, 
                                n_aug=2, top_k=5, 
                                only_label=None)

    # 3. If approach == baseline, we do a custom pipeline
    if args.approach.lower() == "baseline":
        print("Approach: Baseline (fixed embeddings + MLP).")
        # 3A. Convert entire df to embeddings
        embeddings, labels = compute_baseline_embeddings(
            df, 
            text_col=args.text_col, 
            label_col=args.label_col
        )
        # 3B. Create DataLoaders for the baseline
        # We'll do a random split
        from .training.training_utils import baseline_data_split
        train_loader, val_loader = baseline_data_split(
            embeddings, labels, 
            test_size=args.val_split, 
            batch_size=args.batch_size
        )
        # 3C. Train the baseline model
        input_dim = embeddings.shape[1]
        model = BaselineClassifier(input_dim=input_dim, hidden_dims=[256,128])
        train_baseline(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)
        # Optionally, save the baseline model
        torch.save(model.state_dict(), os.path.join(args.output_dir, "baseline_model.pth"))
        print(f"Baseline model saved to {args.output_dir}")
    else:
        print(f"Approach: {args.approach}, model checkpoint: {args.approach_model_name}")
        # 3D. Convert df => HF dataset
        train_ds, val_ds = create_hf_dataset(
            df, 
            text_col=args.text_col, 
            label_col=args.label_col, 
            val_split=args.val_split
        )
        # 3E. Use HF Trainer
        model = get_transformer_classifier(args.approach_model_name, num_labels=2)
        if args.freeze_layers > 0:
            model = freeze_early_layers(model, args.freeze_layers)

        # 3F. Run hugging face trainer
        run_hf_trainer(
            model, train_ds, val_ds, 
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            output_dir=args.output_dir,
            text_col=args.text_col,  # needed if we re-tokenize
            label_col=args.label_col,
            model_name=args.approach_model_name
        )

        print(f"Finished training. Model + tokenizer in {args.output_dir}")
