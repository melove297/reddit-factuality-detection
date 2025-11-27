"""
Model evaluation utilities for computing metrics and visualizing results.

This module provides functions for computing classification metrics,
generating confusion matrices, and saving evaluation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import json


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for ROC-AUC)
        class_names: Names of classes (optional)

    Returns:
        Dictionary containing all computed metrics

    Examples:
        >>> metrics = compute_metrics(y_true, y_pred, y_prob)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        >>> print(f"Macro F1: {metrics['macro_f1']:.4f}")
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Build metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'num_samples': len(y_true),
        'num_correct': int((y_true == y_pred).sum()),
        'confusion_matrix': cm.tolist(),
    }

    # Add per-class metrics
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if class_names is None:
        class_names = [f'class_{i}' for i in unique_labels]

    per_class_metrics = {}
    for i, label in enumerate(unique_labels):
        if i < len(precision):
            per_class_metrics[class_names[i]] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }

    metrics['per_class'] = per_class_metrics

    # ROC-AUC for binary classification
    if y_prob is not None and len(unique_labels) == 2:
        try:
            # For binary classification, use probability of positive class
            if y_prob.ndim == 2:
                y_prob_binary = y_prob[:, 1]
            else:
                y_prob_binary = y_prob

            roc_auc = roc_auc_score(y_true, y_prob_binary)
            metrics['roc_auc'] = float(roc_auc)
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")

    return metrics


def print_metrics(metrics: Dict[str, Any], title: str = "Evaluation Metrics"):
    """
    Print metrics in a formatted, readable way.

    Args:
        metrics: Dictionary of metrics from compute_metrics()
        title: Title for the metrics display
    """
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70)

    print(f"\nOverall Performance:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")

    print(f"\nSample Counts:")
    print(f"  Total:       {metrics['num_samples']}")
    print(f"  Correct:     {metrics['num_correct']}")
    print(f"  Incorrect:   {metrics['num_samples'] - metrics['num_correct']}")

    print(f"\nPer-Class Performance:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"\n  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-Score:  {class_metrics['f1']:.4f}")
        print(f"    Support:   {class_metrics['support']}")

    print("\n" + "="*70 + "\n")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: str,
    normalize: bool = False,
    figsize: tuple = (8, 6)
):
    """
    Plot and save confusion matrix as a heatmap.

    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names for labels
        output_path: Path to save the plot
        normalize: Whether to normalize by true label counts
        figsize: Figure size (width, height)

    Examples:
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> plot_confusion_matrix(cm, ['Non-Factual', 'Factual'], 'results/cm.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize if requested
    if normalize:
        cm_display = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = confusion_matrix
        fmt = 'd'
        title = 'Confusion Matrix'

    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")


def save_metrics_to_json(metrics: Dict[str, Any], output_path: str):
    """
    Save metrics dictionary to JSON file.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    # Convert numpy types to native Python types
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

    metrics_converted = convert_types(metrics)

    with open(output_path, 'w') as f:
        json.dump(metrics_converted, f, indent=2)

    print(f"Metrics saved to: {output_path}")


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str
):
    """
    Generate and save sklearn classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the report (text file)
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    with open(output_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("="*70 + "\n\n")
        f.write(report)

    print(f"Classification report saved to: {output_path}")

    # Also print to console
    print("\nClassification Report:")
    print(report)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    class_name: str = 'Positive Class',
    figsize: tuple = (8, 6)
):
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        output_path: Path to save the plot
        class_name: Name of the positive class
        figsize: Figure size
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {class_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC curve saved to: {output_path}")


def compare_models(
    model_metrics: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple models and create a summary table.

    Args:
        model_metrics: Dictionary mapping model names to their metrics
        output_path: Optional path to save comparison table as CSV

    Returns:
        DataFrame with model comparison

    Examples:
        >>> model_metrics = {
        ...     'Logistic Regression': logreg_metrics,
        ...     'DistilBERT': bert_metrics
        ... }
        >>> comparison = compare_models(model_metrics, 'results/model_comparison.csv')
    """
    comparison_data = []

    for model_name, metrics in model_metrics.items():
        row = {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Macro F1': metrics['macro_f1'],
            'Weighted F1': metrics['weighted_f1'],
        }

        # Add ROC-AUC if available
        if 'roc_auc' in metrics:
            row['ROC-AUC'] = metrics['roc_auc']

        # Add per-class F1 scores
        for class_name, class_metrics in metrics['per_class'].items():
            row[f'{class_name} F1'] = class_metrics['f1']
            row[f'{class_name} Precision'] = class_metrics['precision']
            row[f'{class_name} Recall'] = class_metrics['recall']

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by macro F1 (descending)
    comparison_df = comparison_df.sort_values('Macro F1', ascending=False)

    if output_path:
        comparison_df.to_csv(output_path, index=False)
        print(f"Model comparison saved to: {output_path}")

    return comparison_df


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_metrics: Optional[List[float]] = None,
    metric_name: str = 'F1',
    output_path: str = 'training_curves.png',
    figsize: tuple = (12, 5)
):
    """
    Plot training and validation losses, and optionally validation metrics.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_metrics: Optional list of validation metrics per epoch
        metric_name: Name of the validation metric
        output_path: Path to save the plot
        figsize: Figure size
    """
    epochs = range(1, len(train_losses) + 1)

    if val_metrics is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # Plot losses
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot metrics if provided
    if val_metrics is not None:
        ax2.plot(epochs, val_metrics, 'g-^', label=f'Validation {metric_name}', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel(metric_name, fontsize=12)
        ax2.set_title(f'Validation {metric_name}', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to: {output_path}")


def analyze_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    text_col: str = 'text_clean',
    n_examples: int = 10
) -> pd.DataFrame:
    """
    Analyze misclassified examples.

    Args:
        df: Original DataFrame with text
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        text_col: Name of text column
        n_examples: Number of examples to extract per error type

    Returns:
        DataFrame with error analysis
    """
    # Create analysis DataFrame
    analysis_df = df.copy()
    analysis_df['true_label'] = y_true
    analysis_df['pred_label'] = y_pred
    analysis_df['correct'] = (y_true == y_pred)

    if y_prob is not None:
        if y_prob.ndim == 2:
            analysis_df['confidence'] = y_prob.max(axis=1)
            analysis_df['prob_class_0'] = y_prob[:, 0]
            analysis_df['prob_class_1'] = y_prob[:, 1] if y_prob.shape[1] > 1 else 0
        else:
            analysis_df['confidence'] = y_prob

    # Get misclassified examples
    errors = analysis_df[~analysis_df['correct']].copy()

    print(f"\nError Analysis:")
    print(f"Total errors: {len(errors)} / {len(analysis_df)} ({len(errors)/len(analysis_df)*100:.2f}%)")

    # Analyze by error type
    for true_label in sorted(analysis_df['true_label'].unique()):
        for pred_label in sorted(analysis_df['pred_label'].unique()):
            if true_label != pred_label:
                error_type = errors[
                    (errors['true_label'] == true_label) &
                    (errors['pred_label'] == pred_label)
                ]
                if len(error_type) > 0:
                    print(f"  True={true_label} -> Pred={pred_label}: {len(error_type)} errors")

    return errors


def save_error_examples(
    errors_df: pd.DataFrame,
    output_path: str,
    text_col: str = 'text_clean',
    n_examples: int = 10
):
    """
    Save examples of misclassified instances to a text file.

    Args:
        errors_df: DataFrame of errors from analyze_errors()
        output_path: Path to save examples
        text_col: Name of text column
        n_examples: Number of examples to save
    """
    with open(output_path, 'w') as f:
        f.write("Misclassification Examples\n")
        f.write("="*80 + "\n\n")

        # Sample examples if there are too many
        if len(errors_df) > n_examples:
            examples = errors_df.sample(n=n_examples, random_state=42)
        else:
            examples = errors_df

        for idx, row in examples.iterrows():
            f.write(f"Example {idx}:\n")
            f.write(f"  True Label: {row['true_label']}\n")
            f.write(f"  Predicted Label: {row['pred_label']}\n")

            if 'confidence' in row:
                f.write(f"  Confidence: {row['confidence']:.4f}\n")

            if text_col in row:
                text = str(row[text_col])[:500]  # Truncate long texts
                f.write(f"  Text: {text}\n")

            f.write("\n" + "-"*80 + "\n\n")

    print(f"Error examples saved to: {output_path}")
