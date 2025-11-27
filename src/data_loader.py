"""
Data loading utilities for FACTOID Reddit dataset and Reuters news articles.

This module provides functions to load and perform initial cleaning of the
FACTOID dataset and Reuters corpus.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings


def load_factoid(path: str) -> pd.DataFrame:
    """
    Load the FACTOID dataset from CSV file.

    This function loads Reddit posts from the FACTOID dataset, performs initial
    data quality checks, and normalizes the factuality labels to binary numeric format.

    Args:
        path: Path to the FACTOID CSV file. Expected columns:
              - post_id: unique identifier
              - text: text content of the Reddit post
              - subreddit: subreddit name
              - upvotes: integer upvote count
              - num_comments: integer number of comments
              - factuality_label: categorical label indicating factual reliability

    Returns:
        pd.DataFrame: Cleaned DataFrame with normalized factuality labels (0 or 1).

    Raises:
        FileNotFoundError: If the file at the given path does not exist.
        ValueError: If required columns are missing from the dataset.
    """
    # Load the CSV file
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"FACTOID dataset not found at path: {path}\n"
            "Please ensure the file exists and the path is correct."
        )

    # Verify required columns exist
    required_cols = ['post_id', 'text', 'subreddit', 'upvotes',
                     'num_comments', 'factuality_label']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns in FACTOID dataset: {missing_cols}\n"
            f"Available columns: {list(df.columns)}\n"
            "Please ensure the CSV file has the correct format."
        )

    # Store original size
    original_size = len(df)

    # Drop rows with missing text or labels
    df = df.dropna(subset=['text', 'factuality_label'])

    # Drop rows with empty text
    df = df[df['text'].str.strip().str.len() > 0]

    # Report dropped rows
    dropped = original_size - len(df)
    if dropped > 0:
        warnings.warn(
            f"Dropped {dropped} rows ({dropped/original_size*100:.2f}%) "
            "due to missing or empty text/labels."
        )

    # Normalize factuality labels to numeric {0, 1}
    df = _normalize_factuality_labels(df)

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    print(f"Loaded FACTOID dataset: {len(df)} posts")
    print(f"Label distribution:\n{df['factuality_label'].value_counts()}")
    print(f"Unique subreddits: {df['subreddit'].nunique()}")

    return df


def load_reuters(path: str) -> pd.DataFrame:
    """
    Load Reuters news articles from CSV file.

    This function loads Reuters articles, concatenates title and body text,
    and performs initial cleaning.

    Args:
        path: Path to the Reuters CSV file. Expected columns:
              - article_id: unique identifier
              - title: article title
              - body_text: article body content

    Returns:
        pd.DataFrame: Cleaned DataFrame with concatenated text column.

    Raises:
        FileNotFoundError: If the file at the given path does not exist.
        ValueError: If required columns are missing from the dataset.
    """
    # Load the CSV file
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reuters dataset not found at path: {path}\n"
            "Please ensure the file exists and the path is correct."
        )

    # Verify required columns exist
    required_cols = ['article_id', 'title', 'body_text']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns in Reuters dataset: {missing_cols}\n"
            f"Available columns: {list(df.columns)}\n"
            "Please ensure the CSV file has the correct format."
        )

    # Store original size
    original_size = len(df)

    # Fill NaN values with empty strings for concatenation
    df['title'] = df['title'].fillna('')
    df['body_text'] = df['body_text'].fillna('')

    # Concatenate title and body_text into a single "text" column
    df['text'] = df['title'].str.strip() + ' ' + df['body_text'].str.strip()
    df['text'] = df['text'].str.strip()

    # Drop rows with empty text (both title and body were empty or whitespace)
    df = df[df['text'].str.len() > 0]

    # Report dropped rows
    dropped = original_size - len(df)
    if dropped > 0:
        warnings.warn(
            f"Dropped {dropped} rows ({dropped/original_size*100:.2f}%) "
            "due to missing or empty text content."
        )

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    print(f"Loaded Reuters dataset: {len(df)} articles")

    return df


def _normalize_factuality_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize factuality labels to binary numeric format {0, 1}.

    This internal function handles various label formats and maps them to
    a consistent binary representation where 1 = factual, 0 = non-factual.

    Args:
        df: DataFrame with 'factuality_label' column

    Returns:
        pd.DataFrame: DataFrame with normalized binary labels

    Raises:
        ValueError: If labels cannot be mapped to binary format
    """
    unique_labels = df['factuality_label'].unique()

    # If already numeric binary, verify and return
    if set(unique_labels).issubset({0, 1, 0.0, 1.0}):
        df['factuality_label'] = df['factuality_label'].astype(int)
        return df

    # If already numeric binary but with NaN
    if all(pd.isna(label) or label in {0, 1, 0.0, 1.0} for label in unique_labels):
        df['factuality_label'] = df['factuality_label'].astype(int)
        return df

    # Define mapping for common label formats
    factual_keywords = ['factual', 'true', 'accurate', 'verified', 'reliable', '1', 1, 1.0]
    nonfactual_keywords = ['non_factual', 'nonfactual', 'false', 'inaccurate',
                          'unverified', 'unreliable', 'fake', 'misinformation', '0', 0, 0.0]

    # Create mapping dictionary
    label_map = {}
    for label in unique_labels:
        label_lower = str(label).lower().strip()

        if label_lower in [str(k).lower() for k in factual_keywords]:
            label_map[label] = 1
        elif label_lower in [str(k).lower() for k in nonfactual_keywords]:
            label_map[label] = 0
        else:
            raise ValueError(
                f"Unable to map factuality label '{label}' to binary format.\n"
                f"Unique labels in dataset: {unique_labels}\n"
                "Please ensure labels are in a recognizable format:\n"
                f"  Factual: {factual_keywords}\n"
                f"  Non-factual: {nonfactual_keywords}"
            )

    # Apply mapping
    df['factuality_label'] = df['factuality_label'].map(label_map)

    print(f"Normalized labels - Mapping: {label_map}")

    return df


def get_dataset_statistics(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
    """
    Compute and return basic statistics about a dataset.

    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset for display

    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        'name': dataset_name,
        'num_samples': len(df),
        'num_columns': len(df.columns),
        'columns': list(df.columns),
    }

    # Text statistics if 'text' column exists
    if 'text' in df.columns:
        text_lengths = df['text'].str.len()
        word_counts = df['text'].str.split().str.len()

        stats['text_stats'] = {
            'mean_length': text_lengths.mean(),
            'median_length': text_lengths.median(),
            'max_length': text_lengths.max(),
            'min_length': text_lengths.min(),
            'mean_words': word_counts.mean(),
            'median_words': word_counts.median(),
        }

    # Label distribution if 'factuality_label' exists
    if 'factuality_label' in df.columns:
        label_counts = df['factuality_label'].value_counts()
        stats['label_distribution'] = label_counts.to_dict()
        stats['label_balance'] = label_counts.min() / label_counts.max()

    # Metadata statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'factuality_label':
            stats[f'{col}_stats'] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
            }

    return stats
