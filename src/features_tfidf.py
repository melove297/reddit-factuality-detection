"""
TF-IDF feature extraction utilities for text classification.

This module provides functions for building and applying TF-IDF vectorization
and combining text features with metadata features.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle


def build_tfidf_vectorizer(
    train_texts: List[str],
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Build and fit a TF-IDF vectorizer on training texts.

    This function creates a TfidfVectorizer with specified parameters and
    fits it on the training data only. The fitted vectorizer can then be
    used to transform train, validation, and test sets.

    Args:
        train_texts: List of training text strings
        max_features: Maximum number of features (vocabulary size)
        ngram_range: Tuple of (min_n, max_n) for n-gram range.
                     (1, 1) = unigrams only
                     (1, 2) = unigrams + bigrams
        min_df: Minimum document frequency for a term to be included
        max_df: Maximum document frequency (ignore terms appearing in > max_df of documents)
        sublinear_tf: Apply sublinear tf scaling (replace tf with 1 + log(tf))

    Returns:
        Fitted TfidfVectorizer object

    Examples:
        >>> vectorizer = build_tfidf_vectorizer(train_texts, max_features=10000)
        >>> train_features = vectorizer.transform(train_texts)
        >>> test_features = vectorizer.transform(test_texts)
    """
    print(f"Building TF-IDF vectorizer...")
    print(f"  Max features: {max_features}")
    print(f"  N-gram range: {ngram_range}")
    print(f"  Min document frequency: {min_df}")
    print(f"  Max document frequency: {max_df}")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents='unicode',
        lowercase=True,
        analyzer='word',
        token_pattern=r'\b[a-z]{2,}\b',  # Words with at least 2 letters
        stop_words='english',
        dtype=np.float32
    )

    print(f"Fitting vectorizer on {len(train_texts)} training texts...")
    vectorizer.fit(train_texts)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Number of features: {len(vectorizer.get_feature_names_out())}")

    return vectorizer


def transform_tfidf(vectorizer: TfidfVectorizer, texts: List[str]) -> sparse.csr_matrix:
    """
    Transform texts into TF-IDF feature vectors.

    Args:
        vectorizer: Fitted TfidfVectorizer object
        texts: List of text strings to transform

    Returns:
        Sparse matrix of TF-IDF features with shape (n_texts, n_features)

    Examples:
        >>> test_features = transform_tfidf(vectorizer, test_texts)
        >>> print(test_features.shape)
        (1000, 50000)
    """
    print(f"Transforming {len(texts)} texts to TF-IDF features...")

    features = vectorizer.transform(texts)

    print(f"Feature matrix shape: {features.shape}")
    print(f"Sparsity: {(1.0 - features.nnz / (features.shape[0] * features.shape[1]))*100:.2f}%")

    return features


def concat_features(
    text_features: sparse.csr_matrix,
    metadata_features: np.ndarray
) -> sparse.csr_matrix:
    """
    Concatenate TF-IDF text features with metadata features.

    This function combines sparse TF-IDF features with dense metadata features
    by converting metadata to sparse format and using horizontal stacking.

    Args:
        text_features: Sparse matrix of TF-IDF features (n_samples, n_text_features)
        metadata_features: Dense array of metadata features (n_samples, n_metadata_features)

    Returns:
        Sparse matrix of combined features (n_samples, n_text_features + n_metadata_features)

    Raises:
        ValueError: If number of samples doesn't match between feature sets

    Examples:
        >>> combined = concat_features(tfidf_features, metadata_array)
        >>> print(combined.shape)
        (1000, 50003)  # 50000 TF-IDF + 3 metadata features
    """
    if text_features.shape[0] != metadata_features.shape[0]:
        raise ValueError(
            f"Sample count mismatch: text_features has {text_features.shape[0]} samples "
            f"but metadata_features has {metadata_features.shape[0]} samples"
        )

    # Convert metadata to sparse matrix if it's dense
    if not sparse.issparse(metadata_features):
        metadata_features = sparse.csr_matrix(metadata_features)

    # Horizontally stack the features
    combined_features = sparse.hstack([text_features, metadata_features], format='csr')

    print(f"Combined features shape: {combined_features.shape}")
    print(f"  Text features: {text_features.shape[1]}")
    print(f"  Metadata features: {metadata_features.shape[1]}")

    return combined_features


def get_top_features(
    vectorizer: TfidfVectorizer,
    feature_importance: np.ndarray,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Extract top features by importance (e.g., logistic regression coefficients).

    This function is useful for interpreting TF-IDF + logistic regression models
    by identifying which n-grams have the highest weights.

    Args:
        vectorizer: Fitted TfidfVectorizer object
        feature_importance: Array of feature importances/coefficients
        top_n: Number of top features to extract for each direction

    Returns:
        DataFrame with columns: feature_name, importance, rank

    Examples:
        >>> # After training logistic regression
        >>> top_features = get_top_features(vectorizer, model.coef_[0], top_n=50)
        >>> print(top_features.head())
    """
    feature_names = vectorizer.get_feature_names_out()

    # Handle multi-dimensional importance arrays (take first dimension for binary)
    if feature_importance.ndim > 1:
        feature_importance = feature_importance[0]

    # Only consider TF-IDF features (not metadata)
    n_tfidf_features = len(feature_names)
    if len(feature_importance) > n_tfidf_features:
        feature_importance = feature_importance[:n_tfidf_features]

    # Get top positive features
    top_positive_idx = np.argsort(feature_importance)[-top_n:][::-1]
    top_positive = pd.DataFrame({
        'feature_name': feature_names[top_positive_idx],
        'importance': feature_importance[top_positive_idx],
        'direction': 'positive',
        'rank': range(1, len(top_positive_idx) + 1)
    })

    # Get top negative features
    top_negative_idx = np.argsort(feature_importance)[:top_n]
    top_negative = pd.DataFrame({
        'feature_name': feature_names[top_negative_idx],
        'importance': feature_importance[top_negative_idx],
        'direction': 'negative',
        'rank': range(1, len(top_negative_idx) + 1)
    })

    # Combine
    top_features = pd.concat([top_positive, top_negative], ignore_index=True)

    return top_features


def save_vectorizer(vectorizer: TfidfVectorizer, path: str):
    """
    Save fitted TF-IDF vectorizer to disk.

    Args:
        vectorizer: Fitted TfidfVectorizer to save
        path: Output file path (should end in .pkl)

    Examples:
        >>> save_vectorizer(vectorizer, 'results/tfidf_vectorizer.pkl')
    """
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"TF-IDF vectorizer saved to: {path}")


def load_vectorizer(path: str) -> TfidfVectorizer:
    """
    Load fitted TF-IDF vectorizer from disk.

    Args:
        path: Path to saved vectorizer file

    Returns:
        Loaded TfidfVectorizer object

    Examples:
        >>> vectorizer = load_vectorizer('results/tfidf_vectorizer.pkl')
    """
    with open(path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"TF-IDF vectorizer loaded from: {path}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer


def analyze_vocabulary(vectorizer: TfidfVectorizer, top_n: int = 20) -> Dict:
    """
    Analyze the learned vocabulary of a fitted TF-IDF vectorizer.

    Args:
        vectorizer: Fitted TfidfVectorizer
        top_n: Number of terms to show by IDF score

    Returns:
        Dictionary with vocabulary statistics
    """
    feature_names = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_

    # Get terms with highest IDF (rarest terms)
    highest_idf_idx = np.argsort(idf_scores)[-top_n:][::-1]
    highest_idf_terms = [(feature_names[i], idf_scores[i]) for i in highest_idf_idx]

    # Get terms with lowest IDF (most common terms)
    lowest_idf_idx = np.argsort(idf_scores)[:top_n]
    lowest_idf_terms = [(feature_names[i], idf_scores[i]) for i in lowest_idf_idx]

    # Count n-gram types
    ngram_counts = {1: 0, 2: 0, 3: 0}
    for term in feature_names:
        n = len(term.split())
        if n in ngram_counts:
            ngram_counts[n] += 1
        else:
            ngram_counts[n] = 1

    stats = {
        'vocab_size': len(feature_names),
        'ngram_distribution': ngram_counts,
        'mean_idf': float(np.mean(idf_scores)),
        'std_idf': float(np.std(idf_scores)),
        'highest_idf_terms': highest_idf_terms,
        'lowest_idf_terms': lowest_idf_terms,
    }

    return stats


def print_vocabulary_stats(vectorizer: TfidfVectorizer):
    """
    Print vocabulary statistics in a formatted way.

    Args:
        vectorizer: Fitted TfidfVectorizer
    """
    stats = analyze_vocabulary(vectorizer)

    print("\n" + "="*70)
    print("TF-IDF Vocabulary Statistics")
    print("="*70)
    print(f"Total vocabulary size: {stats['vocab_size']}")
    print(f"\nN-gram distribution:")
    for n, count in sorted(stats['ngram_distribution'].items()):
        print(f"  {n}-grams: {count} ({count/stats['vocab_size']*100:.1f}%)")

    print(f"\nIDF scores:")
    print(f"  Mean: {stats['mean_idf']:.3f}")
    print(f"  Std: {stats['std_idf']:.3f}")

    print(f"\nHighest IDF terms (rarest):")
    for term, score in stats['highest_idf_terms'][:10]:
        print(f"  {term}: {score:.3f}")

    print(f"\nLowest IDF terms (most common):")
    for term, score in stats['lowest_idf_terms'][:10]:
        print(f"  {term}: {score:.3f}")

    print("="*70 + "\n")
