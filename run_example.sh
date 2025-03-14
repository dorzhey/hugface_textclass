#!/usr/bin/env bash
# run_example.sh

# Example usage of hugface_textclass
python -m hugface_textclass \
  --approach distilbert \
  --train_file /path/to/my_dataset.csv \
  --text_col Tweet \
  --label_col Label \
  --clean_text \
  --augment \
  --val_split 0.1 \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --freeze_layers 2 \
  --output_dir ./outputs_distilbert