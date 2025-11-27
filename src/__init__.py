"""
FACTOID Reddit Factuality Classification Project

This package provides a complete implementation for training and evaluating
factuality classifiers on Reddit posts using the FACTOID dataset and comparing
predictions against Reuters news articles for external validation.
"""

__version__ = '1.0.0'
__author__ = 'Ria Kapoor, Bitanya Kebede, Charlie King, George Wright'

# Import key classes and functions for easier access
from .data_loader import load_factoid, load_reuters
from .preprocess import clean_text, apply_cleaning, train_val_test_split
from .model_logreg import FactoidLogRegClassifier
from .evaluate_models import compute_metrics, print_metrics, plot_confusion_matrix
from .utils import set_random_seeds, create_output_dir

# Conditionally import DistilBERT (requires PyTorch)
try:
    from .model_distilbert import DistilBertFactualityClassifier, load_tokenizer
    _has_torch = True
except ImportError:
    _has_torch = False
    DistilBertFactualityClassifier = None
    load_tokenizer = None

__all__ = [
    'load_factoid',
    'load_reuters',
    'clean_text',
    'apply_cleaning',
    'train_val_test_split',
    'FactoidLogRegClassifier',
    'DistilBertFactualityClassifier',
    'load_tokenizer',
    'compute_metrics',
    'print_metrics',
    'plot_confusion_matrix',
    'set_random_seeds',
    'create_output_dir',
]
