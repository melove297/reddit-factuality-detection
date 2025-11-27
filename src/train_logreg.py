"""
Training script for TF-IDF + Logistic Regression baseline model.

This script trains a logistic regression classifier on TF-IDF features
extracted from Reddit posts in the FACTOID dataset.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_factoid
from preprocess import apply_cleaning, train_val_test_split, remove_empty_texts
from features_tfidf import (
    build_tfidf_vectorizer,
    transform_tfidf,
    concat_features,
    get_top_features,
    save_vectorizer,
    print_vocabulary_stats
)
from model_logreg import FactoidLogRegClassifier
from evaluate_models import (
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
    save_metrics_to_json,
    save_classification_report,
    plot_roc_curve
)
from utils import (
    normalize_metadata,
    save_predictions,
    set_random_seeds,
    create_output_dir,
    print_section
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train TF-IDF + Logistic Regression baseline for factuality classification'
    )

    # Data arguments
    parser.add_argument(
        '--factoid_path',
        type=str,
        required=True,
        help='Path to FACTOID dataset CSV file'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/logreg',
        help='Directory to save results and models'
    )

    # Feature engineering arguments
    parser.add_argument(
        '--max_features',
        type=int,
        default=50000,
        help='Maximum number of TF-IDF features'
    )

    parser.add_argument(
        '--ngram_range',
        type=int,
        nargs=2,
        default=[1, 2],
        help='N-gram range for TF-IDF (e.g., 1 2 for unigrams+bigrams)'
    )

    parser.add_argument(
        '--use_metadata',
        action='store_true',
        default=True,
        help='Include metadata features (upvotes, num_comments, subreddit)'
    )

    # Model arguments
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='Inverse regularization strength for logistic regression'
    )

    parser.add_argument(
        '--penalty',
        type=str,
        default='l2',
        choices=['l1', 'l2'],
        help='Regularization penalty'
    )

    parser.add_argument(
        '--class_weight',
        type=str,
        default='balanced',
        choices=['balanced', 'None'],
        help='Class weighting strategy'
    )

    # Data split arguments
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.15,
        help='Proportion of data for test set'
    )

    parser.add_argument(
        '--val_size',
        type=float,
        default=0.15,
        help='Proportion of data for validation set'
    )

    # Other arguments
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--top_n_features',
        type=int,
        default=50,
        help='Number of top features to extract and save'
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    # Set random seeds
    set_random_seeds(args.random_seed)

    # Create output directory
    create_output_dir(args.output_dir)

    # =========================================================================
    # STEP 1: Load and preprocess data
    # =========================================================================
    print_section("STEP 1: Loading and Preprocessing Data")

    # Load FACTOID dataset
    print(f"\nLoading FACTOID dataset from: {args.factoid_path}")
    df = load_factoid(args.factoid_path)

    # Apply text cleaning
    print("\nCleaning text...")
    df = apply_cleaning(df, text_col='text')

    # Remove empty texts
    df = remove_empty_texts(df, text_col='text_clean')

    # Split into train/val/test
    print("\nSplitting data...")
    train_df, val_df, test_df = train_val_test_split(
        df,
        label_col='factuality_label',
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_seed,
        stratify=True
    )

    # =========================================================================
    # STEP 2: Build TF-IDF features
    # =========================================================================
    print_section("STEP 2: Building TF-IDF Features")

    # Build TF-IDF vectorizer on training data
    vectorizer = build_tfidf_vectorizer(
        train_texts=train_df['text_clean'].tolist(),
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range),
        min_df=2,
        max_df=0.95
    )

    # Print vocabulary statistics
    print_vocabulary_stats(vectorizer)

    # Transform all splits
    X_train_text = transform_tfidf(vectorizer, train_df['text_clean'].tolist())
    X_val_text = transform_tfidf(vectorizer, val_df['text_clean'].tolist())
    X_test_text = transform_tfidf(vectorizer, test_df['text_clean'].tolist())

    # Save vectorizer
    vectorizer_path = os.path.join(args.output_dir, 'tfidf_vectorizer.pkl')
    save_vectorizer(vectorizer, vectorizer_path)

    # =========================================================================
    # STEP 3: Process metadata features (optional)
    # =========================================================================
    if args.use_metadata:
        print_section("STEP 3: Processing Metadata Features")

        # Normalize metadata
        metadata_train, metadata_processor = normalize_metadata(train_df)
        metadata_val = metadata_processor.transform(val_df)
        metadata_test = metadata_processor.transform(test_df)

        # Save metadata processor
        processor_path = os.path.join(args.output_dir, 'metadata_processor.pkl')
        metadata_processor.save(processor_path)

        # Concatenate text and metadata features
        X_train = concat_features(X_train_text, metadata_train)
        X_val = concat_features(X_val_text, metadata_val)
        X_test = concat_features(X_test_text, metadata_test)

    else:
        print("\nSkipping metadata features...")
        X_train = X_train_text
        X_val = X_val_text
        X_test = X_test_text

    # Extract labels
    y_train = train_df['factuality_label'].values
    y_val = val_df['factuality_label'].values
    y_test = test_df['factuality_label'].values

    # =========================================================================
    # STEP 4: Train logistic regression model
    # =========================================================================
    print_section("STEP 4: Training Logistic Regression Model")

    # Handle class_weight argument
    class_weight = None if args.class_weight == 'None' else args.class_weight

    # Initialize and train model
    model = FactoidLogRegClassifier(
        C=args.C,
        penalty=args.penalty,
        class_weight=class_weight,
        solver='saga',
        max_iter=1000,
        random_state=args.random_seed,
        verbose=1
    )

    model.fit(X_train, y_train, X_val, y_val)

    # Save model
    model_path = os.path.join(args.output_dir, 'logreg_model.pkl')
    model.save(model_path)

    # =========================================================================
    # STEP 5: Evaluate on validation and test sets
    # =========================================================================
    print_section("STEP 5: Model Evaluation")

    class_names = ['Non-Factual', 'Factual']

    # Validation set evaluation
    print("\n--- Validation Set ---")
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)

    val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob, class_names)
    print_metrics(val_metrics, title="Validation Set Metrics")

    # Test set evaluation
    print("\n--- Test Set ---")
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, class_names)
    print_metrics(test_metrics, title="Test Set Metrics")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    save_metrics_to_json(test_metrics, metrics_path)

    val_metrics_path = os.path.join(args.output_dir, 'val_metrics.json')
    save_metrics_to_json(val_metrics, val_metrics_path)

    # Save classification report
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    save_classification_report(y_test, y_test_pred, class_names, report_path)

    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    cm = np.array(test_metrics['confusion_matrix'])
    plot_confusion_matrix(cm, class_names, cm_path, normalize=False)

    cm_norm_path = os.path.join(args.output_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(cm, class_names, cm_norm_path, normalize=True)

    # Plot ROC curve
    if 'roc_auc' in test_metrics:
        roc_path = os.path.join(args.output_dir, 'roc_curve.png')
        plot_roc_curve(y_test, y_test_prob[:, 1], roc_path, class_name='Factual')

    # Save predictions
    pred_path = os.path.join(args.output_dir, 'test_predictions.csv')
    save_predictions(
        predictions=y_test_pred,
        probabilities=y_test_prob,
        output_path=pred_path,
        post_ids=test_df['post_id'].values if 'post_id' in test_df.columns else None,
        true_labels=y_test
    )

    # =========================================================================
    # STEP 6: Extract and save top features
    # =========================================================================
    print_section("STEP 6: Feature Importance Analysis")

    # Get feature names from vectorizer
    feature_names = list(vectorizer.get_feature_names_out())

    # Add metadata feature names if used
    if args.use_metadata:
        feature_names.extend(metadata_processor.get_feature_names())

    # Print top features
    model.print_top_features(feature_names, top_n=args.top_n_features)

    # Get and save top features to CSV
    top_pos, top_neg = model.get_top_features(feature_names, top_n=args.top_n_features)

    # Create DataFrame
    top_features_df = pd.DataFrame({
        'feature': [f[0] for f in top_pos + top_neg],
        'coefficient': [f[1] for f in top_pos + top_neg],
        'direction': ['positive'] * len(top_pos) + ['negative'] * len(top_neg),
        'abs_coefficient': [abs(f[1]) for f in top_pos + top_neg]
    })

    top_features_df = top_features_df.sort_values('abs_coefficient', ascending=False)

    features_path = os.path.join(args.output_dir, 'top_features.csv')
    top_features_df.to_csv(features_path, index=False)
    print(f"\nTop features saved to: {features_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Training Complete!")

    print(f"\nResults saved to: {args.output_dir}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1:  {test_metrics['macro_f1']:.4f}")
    if 'roc_auc' in test_metrics:
        print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    print("\nFiles saved:")
    print(f"  - Model: {model_path}")
    print(f"  - Vectorizer: {vectorizer_path}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Confusion Matrix: {cm_path}")
    print(f"  - Top Features: {features_path}")
    print(f"  - Predictions: {pred_path}")


if __name__ == '__main__':
    main()
