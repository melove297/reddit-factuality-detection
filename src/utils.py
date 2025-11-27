"""
Utility functions for metadata processing and general helper functions.

This module provides functions for normalizing and encoding metadata features,
as well as general utility functions for saving/loading models and results.
"""

import numpy as np
import pandas as pd
import pickle
import json
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import warnings


class MetadataProcessor:
    """
    Process and normalize metadata features for machine learning models.

    This class handles normalization of numeric metadata (upvotes, num_comments)
    and encoding of categorical metadata (subreddit) into feature vectors.
    """

    def __init__(self, numeric_cols: List[str] = None, categorical_cols: List[str] = None):
        """
        Initialize the MetadataProcessor.

        Args:
            numeric_cols: List of numeric column names to normalize
            categorical_cols: List of categorical column names to encode
        """
        self.numeric_cols = numeric_cols or ['upvotes', 'num_comments']
        self.categorical_cols = categorical_cols or ['subreddit']

        self.scaler = StandardScaler()
        self.cat_encoders = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'MetadataProcessor':
        """
        Fit the processor on training data.

        Args:
            df: Training DataFrame containing metadata columns

        Returns:
            Self (for method chaining)
        """
        # Fit numeric scaler
        numeric_data = self._get_numeric_data(df)
        if numeric_data.shape[1] > 0:
            self.scaler.fit(numeric_data)

        # Fit categorical encoders
        for col in self.categorical_cols:
            if col in df.columns:
                encoder = LabelEncoder()
                # Fill NaN with a placeholder
                values = df[col].fillna('UNKNOWN').astype(str)
                encoder.fit(values)
                self.cat_encoders[col] = encoder

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform metadata into normalized feature matrix.

        Args:
            df: DataFrame containing metadata columns

        Returns:
            Numpy array of shape (n_samples, n_features)

        Raises:
            RuntimeError: If transform is called before fit
        """
        if not self.is_fitted:
            raise RuntimeError("MetadataProcessor must be fitted before transform")

        features = []

        # Transform numeric features
        numeric_data = self._get_numeric_data(df)
        if numeric_data.shape[1] > 0:
            numeric_features = self.scaler.transform(numeric_data)
            features.append(numeric_features)

        # Transform categorical features
        for col in self.categorical_cols:
            if col in df.columns:
                values = df[col].fillna('UNKNOWN').astype(str)
                # Handle unseen categories by mapping to -1
                encoded = np.array([
                    self.cat_encoders[col].transform([v])[0]
                    if v in self.cat_encoders[col].classes_ else -1
                    for v in values
                ]).reshape(-1, 1)
                features.append(encoded)

        # Concatenate all features
        if len(features) == 0:
            return np.zeros((len(df), 0))

        return np.hstack(features)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the processor and transform data in one step.

        Args:
            df: DataFrame containing metadata columns

        Returns:
            Numpy array of transformed features
        """
        return self.fit(df).transform(df)

    def _get_numeric_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and prepare numeric columns from DataFrame.

        Args:
            df: DataFrame containing metadata

        Returns:
            Numpy array of numeric features
        """
        available_cols = [col for col in self.numeric_cols if col in df.columns]

        if not available_cols:
            return np.zeros((len(df), 0))

        # Fill NaN with median or 0
        numeric_data = df[available_cols].fillna(0).values

        return numeric_data

    def get_feature_names(self) -> List[str]:
        """
        Get names of all features after transformation.

        Returns:
            List of feature names
        """
        names = []

        # Add numeric feature names
        names.extend(self.numeric_cols)

        # Add categorical feature names
        names.extend(self.categorical_cols)

        return names

    def save(self, path: str):
        """Save the fitted processor to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'MetadataProcessor':
        """Load a fitted processor from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def normalize_metadata(
    df: pd.DataFrame,
    numeric_cols: List[str] = None,
    categorical_cols: List[str] = None
) -> Tuple[np.ndarray, MetadataProcessor]:
    """
    Normalize metadata features using z-score normalization and encoding.

    This is a convenience function that creates, fits, and transforms in one call.

    Args:
        df: DataFrame containing metadata columns
        numeric_cols: List of numeric columns to normalize (default: ['upvotes', 'num_comments'])
        categorical_cols: List of categorical columns to encode (default: ['subreddit'])

    Returns:
        Tuple of (feature_matrix, fitted_processor)
            - feature_matrix: numpy array of shape (n_samples, n_features)
            - fitted_processor: MetadataProcessor object for transforming new data

    Example:
        >>> features, processor = normalize_metadata(train_df)
        >>> # Later, transform validation data
        >>> val_features = processor.transform(val_df)
    """
    processor = MetadataProcessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    features = processor.fit_transform(df)

    print(f"Metadata features extracted: {features.shape[1]} features from {len(df)} samples")
    print(f"Feature names: {processor.get_feature_names()}")

    return features, processor


def save_predictions(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray],
    output_path: str,
    post_ids: Optional[np.ndarray] = None,
    true_labels: Optional[np.ndarray] = None
):
    """
    Save model predictions to CSV file.

    Args:
        predictions: Array of predicted labels
        probabilities: Array of prediction probabilities (can be None)
        output_path: Path to save CSV file
        post_ids: Optional array of post IDs
        true_labels: Optional array of true labels
    """
    results = pd.DataFrame()

    if post_ids is not None:
        results['post_id'] = post_ids

    results['predicted_label'] = predictions

    if probabilities is not None:
        if probabilities.ndim == 1:
            results['probability'] = probabilities
        else:
            # Multi-class probabilities
            for i in range(probabilities.shape[1]):
                results[f'probability_class_{i}'] = probabilities[:, i]

    if true_labels is not None:
        results['true_label'] = true_labels
        results['correct'] = (predictions == true_labels).astype(int)

    results.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def save_json(data: Dict[str, Any], output_path: str):
    """
    Save dictionary to JSON file with proper formatting.

    Args:
        data: Dictionary to save
        output_path: Path to save JSON file
    """
    # Convert numpy types to Python native types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    data_converted = convert_types(data)

    with open(output_path, 'w') as f:
        json.dump(data_converted, f, indent=2)

    print(f"Data saved to: {output_path}")


def load_json(path: str) -> Dict[str, Any]:
    """
    Load JSON file into dictionary.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary containing loaded data
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def create_output_dir(output_dir: str):
    """
    Create output directory if it doesn't exist.

    Args:
        output_dir: Path to output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary for display.

    Args:
        metrics: Dictionary of metric names to values
        precision: Number of decimal places to display

    Returns:
        Formatted string representation
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.{precision}f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def print_section(title: str, width: int = 70):
    """
    Print a formatted section header.

    Args:
        title: Section title
        width: Width of the separator line
    """
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' to maximize metric, 'min' to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.compare = np.greater if mode == 'max' else np.less
        self.delta = min_delta if mode == 'max' else -min_delta

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.compare(score, self.best_score + self.delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered after {self.counter} epochs without improvement")
                return True

        return False

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
