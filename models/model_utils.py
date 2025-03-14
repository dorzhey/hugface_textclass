# hugface_textclass/models/model_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification,
                          AutoModel)

class BaselineClassifier(nn.Module):
    """
    Simple feed-forward classifier for baseline usage.
    We assume the user pre-computes embeddings for each sample
    and feeds them into this net.
    """
    def __init__(self, input_dim=768, hidden_dims=[256, 128]):
        super(BaselineClassifier, self).__init__()
        layers = []
        in_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = hd
        layers.append(nn.Linear(in_dim, 1))  # Binary
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

def get_transformer_classifier(model_name="distilbert-base-uncased", num_labels=2):
    """
    Returns a transformer classification model ready for training.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model

def freeze_early_layers(model, num_layers_to_freeze=2):
    """
    Example function that can freeze first N layers of a transformer model.
    For DistilBERT, 'model.distilbert.transformer.layer[:num_layers_to_freeze]'
    """
    try:
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        layers = model.distilbert.transformer.layer
        for i in range(num_layers_to_freeze):
            for param in layers[i].parameters():
                param.requires_grad = False
    except AttributeError:
        print("Freeze early layers not implemented for this model type.")
    return model
