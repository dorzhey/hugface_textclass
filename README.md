# hugface_textclass

A Python tool for benchmarking text classification approaches (baseline feed-forward vs. Transformer fine-tuning).  
It handles:
- Automated data loading and optional text cleaning
- Augmentation via nlpaug
- Baseline NN with frozen Transformer embeddings
- Full Transformer finetuning (Hugging Face Trainer)
- Optional freezing of lower Transformer layers

## Installation

```bash
git clone https://github.com/dorzhey/hugface_textclass.git
conda env create -f environment.yml
conda activate hugface_textclass_env
cd hugface_textclass
bash run_example.sh
```
For HPC:
```bash
module load gcc/11.2.0
module load cuda/12.1
```

# Usage example
```bash
python -m hugface_textclass --approach distilbert \
    --train_file data/my_tweets.csv \
    --text_col Tweet --label_col Label \
    --clean_text \
    --augment \
    --val_split 0.1 \
    --epochs 3 \
    --batch_size 16 \
    --lr 2e-5 \
    --freeze_layers 2 \
    --output_dir ./outputs_distilbert
```

This example:

Loads my_tweets.csv, columns Tweet and Label
Cleans text, does augmentation
Splits 10% for validation
Fine-tunes DistilBERT for 3 epochs, freeze first 2 layers
Saves best model to ./outputs_distilbert

For a baseline run:
```bash
python -m hugface_textclass --approach baseline \
    --train_file data/my_tweets.csv \
    --text_col Tweet --label_col Label \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3 \
    --output_dir ./outputs_baseline
```
The tool automatically extracts embeddings for each sample from a pretrained Transformer, then trains a simple feed-forward classifier on those fixed embeddings.