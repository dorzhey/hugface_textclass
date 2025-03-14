# hugface_textclass/models/model_utils.py

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

##################################################
# Baseline MLP
##################################################
class BaselineClassifier(nn.Module):
    """
    Simple feed-forward MLP for the baseline approach,
    expecting an input_dim from pre-computed embeddings.
    """
    def __init__(self, input_dim=768, hidden_dims=[256, 128]):
        super(BaselineClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hd
        # final binary classification
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # shape: [batch, input_dim]
        out = self.net(x)
        return out.squeeze(-1)

##################################################
# Transformer Model
##################################################
def get_transformer_classifier(model_name="distilbert-base-uncased", num_labels=2):
    """
    Returns a pretrained model with a classification head,
    from a given HF model_name.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model

def freeze_early_layers(model, num_layers_to_freeze=2):
    """
    Example for DistilBERT: freeze the embedding + first N layers of the encoder.
    If the model does not have distilbert, handle similarly for BERT or other architectures.
    """
    try:
        # freeze embeddings
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        # freeze the first N layers
        layers = model.distilbert.transformer.layer
        for i in range(num_layers_to_freeze):
            for param in layers[i].parameters():
                param.requires_grad = False
        print(f"Froze first {num_layers_to_freeze} DistilBERT layers.")
    except AttributeError:
        print("Warning: freeze_early_layers not implemented for this model architecture.")
    return model
