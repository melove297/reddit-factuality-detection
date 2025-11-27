"""
DistilBERT-based transformer model for factuality classification.

This module provides a PyTorch model class using DistilBERT from Hugging Face
for post-level factuality classification.
"""

import torch
import torch.nn as nn
from transformers import (
    DistilBertModel,
    DistilBertTokenizerFast,
    DistilBertConfig
)
from typing import Dict, Optional
import warnings


class DistilBertFactualityClassifier(nn.Module):
    """
    DistilBERT-based classifier for Reddit post factuality prediction.

    This model loads the DistilBERT encoder and adds a classification head
    for binary or multi-class factuality classification.
    """

    def __init__(
        self,
        num_labels: int = 2,
        model_name: str = 'distilbert-base-uncased',
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize the DistilBERT classifier.

        Args:
            num_labels: Number of output classes (2 for binary classification)
            model_name: Pretrained DistilBERT model name
            dropout: Dropout probability for classification head
            hidden_dim: Optional hidden layer dimension (if None, use direct classification)
        """
        super().__init__()

        self.num_labels = num_labels
        self.model_name = model_name
        self.hidden_dim = hidden_dim

        # Load pretrained DistilBERT
        print(f"Loading pretrained DistilBERT: {model_name}")
        self.distilbert = DistilBertModel.from_pretrained(model_name)

        # Get hidden size from config
        self.bert_hidden_size = self.distilbert.config.hidden_size

        # Classification head
        if hidden_dim is not None:
            # Two-layer classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.bert_hidden_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels)
            )
        else:
            # Single-layer classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.bert_hidden_size, num_labels)
            )

        print(f"Model initialized:")
        print(f"  DistilBERT hidden size: {self.bert_hidden_size}")
        print(f"  Number of labels: {num_labels}")
        print(f"  Dropout: {dropout}")
        if hidden_dim:
            print(f"  Hidden layer: {hidden_dim}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_length)
            attention_mask: Attention mask tensor of shape (batch_size, seq_length)

        Returns:
            Dictionary containing:
                - 'logits': Unnormalized predictions of shape (batch_size, num_labels)
                - 'hidden_state': Last hidden state from DistilBERT (batch_size, hidden_size)
        """
        # Pass through DistilBERT encoder
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get [CLS] token representation (first token)
        # last_hidden_state shape: (batch_size, seq_length, hidden_size)
        hidden_state = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Pass through classification head
        logits = self.classifier(hidden_state)  # (batch_size, num_labels)

        return {
            'logits': logits,
            'hidden_state': hidden_state
        }

    def freeze_encoder(self):
        """
        Freeze DistilBERT encoder parameters (only train classifier head).
        Useful for faster training when compute is limited.
        """
        for param in self.distilbert.parameters():
            param.requires_grad = False

        print("DistilBERT encoder frozen. Only classifier head will be trained.")

    def unfreeze_encoder(self):
        """Unfreeze DistilBERT encoder parameters."""
        for param in self.distilbert.parameters():
            param.requires_grad = True

        print("DistilBERT encoder unfrozen. All parameters will be trained.")

    def get_num_trainable_params(self) -> int:
        """
        Get the number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        """
        Get the total number of parameters.

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())

    def print_model_info(self):
        """Print model architecture and parameter information."""
        print("\n" + "="*70)
        print("DistilBERT Factuality Classifier Information")
        print("="*70)
        print(f"Model name: {self.model_name}")
        print(f"Number of labels: {self.num_labels}")
        print(f"Total parameters: {self.get_num_total_params():,}")
        print(f"Trainable parameters: {self.get_num_trainable_params():,}")
        print("="*70 + "\n")


def create_distilbert_model(
    num_labels: int = 2,
    model_name: str = 'distilbert-base-uncased',
    dropout: float = 0.1,
    freeze_encoder: bool = False
) -> DistilBertFactualityClassifier:
    """
    Convenience function to create a DistilBERT classifier.

    Args:
        num_labels: Number of output classes
        model_name: Pretrained model name
        dropout: Dropout probability
        freeze_encoder: Whether to freeze encoder weights

    Returns:
        Initialized DistilBertFactualityClassifier
    """
    model = DistilBertFactualityClassifier(
        num_labels=num_labels,
        model_name=model_name,
        dropout=dropout
    )

    if freeze_encoder:
        model.freeze_encoder()

    model.print_model_info()

    return model


def load_tokenizer(model_name: str = 'distilbert-base-uncased') -> DistilBertTokenizerFast:
    """
    Load DistilBERT tokenizer.

    Args:
        model_name: Pretrained model name

    Returns:
        DistilBertTokenizerFast instance
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    print(f"Tokenizer loaded:")
    print(f"  Vocabulary size: {len(tokenizer)}")
    print(f"  Model max length: {tokenizer.model_max_length}")

    return tokenizer


def save_model(
    model: DistilBertFactualityClassifier,
    tokenizer: DistilBertTokenizerFast,
    output_dir: str
):
    """
    Save model and tokenizer to disk.

    Args:
        model: Trained DistilBertFactualityClassifier
        tokenizer: DistilBertTokenizerFast
        output_dir: Directory to save model and tokenizer
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save model state dict
    model_path = os.path.join(output_dir, 'pytorch_model.bin')
    torch.save(model.state_dict(), model_path)

    # Save model config
    config = {
        'num_labels': model.num_labels,
        'model_name': model.model_name,
        'hidden_dim': model.hidden_dim,
    }

    config_path = os.path.join(output_dir, 'model_config.pt')
    torch.save(config, config_path)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to: {output_dir}")


def load_model(
    model_dir: str,
    device: Optional[torch.device] = None
) -> tuple:
    """
    Load model and tokenizer from disk.

    Args:
        model_dir: Directory containing saved model
        device: Device to load model on (if None, uses CPU)

    Returns:
        Tuple of (model, tokenizer)
    """
    import os

    if device is None:
        device = torch.device('cpu')

    # Load config
    config_path = os.path.join(model_dir, 'model_config.pt')
    config = torch.load(config_path, map_location=device)

    # Create model
    model = DistilBertFactualityClassifier(
        num_labels=config['num_labels'],
        model_name=config['model_name'],
        hidden_dim=config.get('hidden_dim', None)
    )

    # Load state dict
    model_path = os.path.join(model_dir, 'pytorch_model.bin')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

    model.to(device)
    model.eval()

    print(f"Model loaded from: {model_dir}")
    print(f"Device: {device}")

    return model, tokenizer


class DistilBertDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for DistilBERT text classification.
    """

    def __init__(
        self,
        texts: list,
        labels: Optional[list] = None,
        tokenizer: DistilBertTokenizerFast = None,
        max_length: int = 256
    ):
        """
        Initialize dataset.

        Args:
            texts: List of text strings
            labels: List of labels (optional, for inference)
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

        # Add label if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def create_data_loader(
    texts: list,
    labels: Optional[list],
    tokenizer: DistilBertTokenizerFast,
    max_length: int = 256,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for DistilBERT training/inference.

    Args:
        texts: List of text strings
        labels: List of labels (can be None for inference)
        tokenizer: DistilBERT tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader instance
    """
    dataset = DistilBertDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader
