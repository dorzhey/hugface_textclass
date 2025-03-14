# hugface_textclass

hugface_textclass is an open-source tool that streamlines text classification by integrating Hugging Face models, PyTorch, and data processing utilities. It enables quick experimentation with various approaches, from baseline feed-forward classifiers to full Transformer fine-tuning. The framework provides automatic dataset handling, optional text augmentation, and flexible training configurations, making it easy to benchmark new datasets efficiently.

## Features
- **Full support for Hugging Face models** (e.g., BERT, DistilBERT, GPT) with the `Trainer` API
- **Baseline approach** using frozen Transformer embeddings with a simple MLP classifier
- **Text augmentation** via `nlpaug` for boosting training diversity
- **Data preprocessing** with automatic text cleaning and tokenization
- **Flexible training setup** (adjustable batch size, learning rate, epochs, layer freezing)
- **Easy-to-use CLI** for specifying models, hyperparameters, and data handling
- **Fully open-source** and extendable for custom modifications

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

Development Notes
The project is structured into three main components: data_utils.py (data handling), model_utils.py (model setup), and training_utils.py (training execution).
cli.py orchestrates argument parsing and model execution.
__main__.py enables execution as a Python module (python -m hugface_textclass).

Contributions are welcome!

This tool is designed to make text classification benchmarking fast and simple by leveraging modern libraries. Whether you're running a quick baseline or fine-tuning a Transformer, hugface_textclass provides a clean and efficient workflow.