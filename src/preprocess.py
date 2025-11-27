"""
Text preprocessing and data splitting utilities.

This module provides functions for cleaning text data and splitting datasets
into train, validation, and test sets with stratification support.
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import warnings


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.

    This function performs the following preprocessing steps:
    1. Lowercase conversion
    2. URL removal
    3. HTML tag removal
    4. Reddit-specific artifact removal (@username, r/subreddit)
    5. Non-ASCII character removal
    6. Excessive whitespace normalization

    Args:
        text: Raw text string to clean

    Returns:
        Cleaned and normalized text string

    Examples:
        >>> clean_text("Check out https://example.com! #AI")
        'check out ai'
        >>> clean_text("Hello @user, visit r/science")
        'hello visit'
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove Reddit-specific artifacts
    # Remove @username mentions
    text = re.sub(r'@\w+', '', text)
    # Remove r/subreddit references
    text = re.sub(r'r/\w+', '', text)
    # Remove u/username references
    text = re.sub(r'u/\w+', '', text)

    # Remove common Reddit artifacts
    text = re.sub(r'\[deleted\]', '', text)
    text = re.sub(r'\[removed\]', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s\.\,\!\?\;\:\-]', ' ', text)

    # Normalize whitespace (collapse multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def apply_cleaning(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """
    Apply text cleaning to a specified column in a DataFrame.

    Creates a new column 'text_clean' with cleaned text while preserving
    the original text column.

    Args:
        df: DataFrame containing text data
        text_col: Name of the column containing text to clean (default: 'text')

    Returns:
        DataFrame with new 'text_clean' column added

    Raises:
        ValueError: If the specified text column does not exist
    """
    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in DataFrame.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Apply cleaning function to each text entry
    print(f"Cleaning text in column '{text_col}'...")
    df['text_clean'] = df[text_col].apply(clean_text)

    # Report statistics
    original_lengths = df[text_col].str.len()
    cleaned_lengths = df['text_clean'].str.len()

    print(f"Original text - Mean length: {original_lengths.mean():.1f} chars")
    print(f"Cleaned text - Mean length: {cleaned_lengths.mean():.1f} chars")
    print(f"Average reduction: {(1 - cleaned_lengths.mean()/original_lengths.mean())*100:.1f}%")

    # Check for empty cleaned texts
    empty_count = (df['text_clean'].str.len() == 0).sum()
    if empty_count > 0:
        warnings.warn(
            f"{empty_count} texts became empty after cleaning. "
            "Consider removing these rows before training."
        )

    return df


def train_val_test_split(
    df: pd.DataFrame,
    label_col: str = 'factuality_label',
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.

    Performs a two-step split to create three partitions with optional
    stratification to maintain label distribution across splits.

    Args:
        df: DataFrame to split
        label_col: Name of the column containing labels for stratification
        test_size: Proportion of data to allocate to test set (default: 0.15)
        val_size: Proportion of data to allocate to validation set (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)
        stratify: Whether to perform stratified split based on labels (default: True)

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        ValueError: If label_col not found when stratify=True
        ValueError: If test_size + val_size >= 1.0

    Examples:
        >>> train, val, test = train_val_test_split(df, test_size=0.15, val_size=0.15)
        >>> # Results in approximately 70/15/15 split
    """
    # Validate split sizes
    if test_size + val_size >= 1.0:
        raise ValueError(
            f"test_size ({test_size}) + val_size ({val_size}) must be < 1.0\n"
            f"Current sum: {test_size + val_size}"
        )

    # Validate label column exists if stratification requested
    if stratify and label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in DataFrame.\n"
            f"Available columns: {list(df.columns)}\n"
            "Set stratify=False to split without stratification."
        )

    # Determine stratification variable
    stratify_var = df[label_col] if stratify else None

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_var
    )

    # Calculate validation size as proportion of remaining data
    val_size_adjusted = val_size / (1 - test_size)

    # Second split: separate validation set from training set
    stratify_var_train = train_val_df[label_col] if stratify else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_var_train
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Report split statistics
    print("\n" + "="*60)
    print("Dataset Split Summary")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Train set: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    if stratify and label_col in df.columns:
        print(f"\nLabel distribution (stratified by '{label_col}'):")
        print("\nTrain set:")
        print(train_df[label_col].value_counts(normalize=True).sort_index())
        print("\nValidation set:")
        print(val_df[label_col].value_counts(normalize=True).sort_index())
        print("\nTest set:")
        print(test_df[label_col].value_counts(normalize=True).sort_index())

    print("="*60 + "\n")

    return train_df, val_df, test_df


def remove_empty_texts(df: pd.DataFrame, text_col: str = 'text_clean') -> pd.DataFrame:
    """
    Remove rows with empty or very short text after cleaning.

    Args:
        df: DataFrame to filter
        text_col: Name of the text column to check (default: 'text_clean')

    Returns:
        DataFrame with empty texts removed
    """
    if text_col not in df.columns:
        warnings.warn(f"Column '{text_col}' not found. Returning original DataFrame.")
        return df

    original_size = len(df)

    # Remove empty or whitespace-only texts
    df = df[df[text_col].str.strip().str.len() > 0].copy()

    # Optionally remove very short texts (less than 3 characters)
    df = df[df[text_col].str.len() >= 3].copy()

    removed = original_size - len(df)
    if removed > 0:
        print(f"Removed {removed} rows with empty or very short text "
              f"({removed/original_size*100:.2f}%)")

    return df.reset_index(drop=True)


def get_text_statistics(df: pd.DataFrame, text_col: str = 'text') -> dict:
    """
    Compute statistics about text lengths and characteristics.

    Args:
        df: DataFrame containing text data
        text_col: Name of the text column to analyze

    Returns:
        Dictionary containing text statistics
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    texts = df[text_col].astype(str)

    # Character counts
    char_counts = texts.str.len()

    # Word counts
    word_counts = texts.str.split().str.len()

    # Sentence counts (approximate by counting periods, !, ?)
    sentence_counts = texts.str.count(r'[.!?]+')

    stats = {
        'num_texts': len(texts),
        'char_count': {
            'mean': char_counts.mean(),
            'median': char_counts.median(),
            'std': char_counts.std(),
            'min': char_counts.min(),
            'max': char_counts.max(),
            'percentile_25': char_counts.quantile(0.25),
            'percentile_75': char_counts.quantile(0.75),
            'percentile_95': char_counts.quantile(0.95),
        },
        'word_count': {
            'mean': word_counts.mean(),
            'median': word_counts.median(),
            'std': word_counts.std(),
            'min': word_counts.min(),
            'max': word_counts.max(),
            'percentile_95': word_counts.quantile(0.95),
        },
        'sentence_count': {
            'mean': sentence_counts.mean(),
            'median': sentence_counts.median(),
        }
    }

    return stats


def balance_dataset(
    df: pd.DataFrame,
    label_col: str = 'factuality_label',
    method: str = 'undersample',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balance dataset by undersampling majority class or oversampling minority class.

    Args:
        df: DataFrame to balance
        label_col: Name of the label column
        method: 'undersample' or 'oversample'
        random_state: Random seed for reproducibility

    Returns:
        Balanced DataFrame
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    label_counts = df[label_col].value_counts()
    minority_class = label_counts.idxmin()
    majority_class = label_counts.idxmax()
    min_count = label_counts.min()
    max_count = label_counts.max()

    print(f"Original class distribution:")
    print(label_counts)

    if method == 'undersample':
        # Undersample majority class
        majority_df = df[df[label_col] == majority_class].sample(
            n=min_count, random_state=random_state
        )
        minority_df = df[df[label_col] == minority_class]
        balanced_df = pd.concat([majority_df, minority_df], ignore_index=True)

    elif method == 'oversample':
        # Oversample minority class
        minority_df = df[df[label_col] == minority_class].sample(
            n=max_count, replace=True, random_state=random_state
        )
        majority_df = df[df[label_col] == majority_class]
        balanced_df = pd.concat([majority_df, minority_df], ignore_index=True)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'undersample' or 'oversample'")

    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\nBalanced class distribution ({method}):")
    print(balanced_df[label_col].value_counts())

    return balanced_df
