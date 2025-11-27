"""
Visualization utilities for research results.

This module provides comprehensive visualization functions for analyzing
model performance, training dynamics, and Reuters alignment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
import os


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_similarity_distributions(
    df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (12, 5)
):
    """
    Plot distribution of Reuters similarity scores by predicted factuality.

    Args:
        df: DataFrame with 'pred_label' and 'max_similarity' columns
        output_path: Path to save the plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Map labels to readable names
    label_map = {0: 'Non-Factual', 1: 'Factual'}
    df_plot = df.copy()
    df_plot['pred_label_name'] = df_plot['pred_label'].map(label_map)

    # Histogram
    for label in sorted(df_plot['pred_label'].unique()):
        label_name = label_map[label]
        subset = df_plot[df_plot['pred_label'] == label]

        ax1.hist(
            subset['max_similarity'],
            bins=50,
            alpha=0.6,
            label=label_name,
            edgecolor='black'
        )

    ax1.set_xlabel('Max Similarity to Reuters', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Reuters Similarity\nby Predicted Label', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # KDE plot
    for label in sorted(df_plot['pred_label'].unique()):
        label_name = label_map[label]
        subset = df_plot[df_plot['pred_label'] == label]

        subset['max_similarity'].plot.kde(
            ax=ax2,
            label=label_name,
            linewidth=2
        )

    ax2.set_xlabel('Max Similarity to Reuters', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Density of Reuters Similarity\nby Predicted Label', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Similarity distributions plot saved to: {output_path}")


def plot_probability_vs_similarity_scatter(
    df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (10, 8)
):
    """
    Create scatter plot of factuality probability vs Reuters similarity.

    Args:
        df: DataFrame with prediction probabilities and similarity scores
        output_path: Path to save the plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color by true label if available
    if 'factuality_label' in df.columns:
        colors = df['factuality_label'].map({0: 'red', 1: 'blue'})
        labels_for_legend = df['factuality_label'].map({0: 'True: Non-Factual', 1: 'True: Factual'})

        # Plot each class separately for legend
        for label, color, name in [(0, 'red', 'True: Non-Factual'), (1, 'blue', 'True: Factual')]:
            subset = df[df['factuality_label'] == label]
            ax.scatter(
                subset['prob_factual'],
                subset['max_similarity'],
                c=color,
                alpha=0.5,
                s=20,
                label=name,
                edgecolors='black',
                linewidth=0.5
            )
    else:
        # Color by predicted label
        colors = df['pred_label'].map({0: 'red', 1: 'blue'})
        ax.scatter(
            df['prob_factual'],
            df['max_similarity'],
            c=colors,
            alpha=0.5,
            s=20,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel('Predicted Probability (Factual)', fontsize=12)
    ax.set_ylabel('Max Similarity to Reuters', fontsize=12)
    ax.set_title('Factuality Probability vs Reuters Similarity', fontsize=14, fontweight='bold')

    if 'factuality_label' in df.columns:
        ax.legend(fontsize=10, markerscale=2)

    ax.grid(True, alpha=0.3)

    # Add correlation text
    correlation = df['prob_factual'].corr(df['max_similarity'])
    ax.text(
        0.05, 0.95,
        f'Correlation: {correlation:.3f}',
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Probability vs similarity scatter plot saved to: {output_path}")


def plot_subreddit_factuality(
    df: pd.DataFrame,
    output_path: str,
    top_n: int = 20,
    figsize: tuple = (12, 8)
):
    """
    Plot factuality rates by subreddit.

    Args:
        df: DataFrame with 'subreddit' and 'pred_label' columns
        output_path: Path to save the plot
        top_n: Number of top subreddits to display
        figsize: Figure size
    """
    if 'subreddit' not in df.columns:
        print("Warning: 'subreddit' column not found. Skipping subreddit analysis.")
        return

    # Aggregate by subreddit
    subreddit_stats = df.groupby('subreddit').agg({
        'pred_label': ['count', 'mean']
    }).reset_index()

    subreddit_stats.columns = ['subreddit', 'count', 'factual_rate']

    # Filter to subreddits with minimum posts
    min_posts = 10
    subreddit_stats = subreddit_stats[subreddit_stats['count'] >= min_posts]

    # Sort by factual rate and get top/bottom N
    subreddit_stats = subreddit_stats.sort_values('factual_rate', ascending=True)

    # Get top N with lowest factual rate (most misinformation)
    bottom_n = subreddit_stats.head(top_n // 2)

    # Get top N with highest factual rate
    top_n_subs = subreddit_stats.tail(top_n // 2)

    # Combine
    plot_df = pd.concat([bottom_n, top_n_subs])

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['red' if rate < 0.5 else 'green' for rate in plot_df['factual_rate']]

    bars = ax.barh(
        range(len(plot_df)),
        plot_df['factual_rate'],
        color=colors,
        alpha=0.7,
        edgecolor='black'
    )

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['subreddit'], fontsize=9)
    ax.set_xlabel('Predicted Factual Rate', fontsize=12)
    ax.set_ylabel('Subreddit', fontsize=12)
    ax.set_title(f'Factuality Rates by Subreddit (Top {top_n})', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])

    # Add count annotations
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(
            row['factual_rate'] + 0.02,
            i,
            f"n={int(row['count'])}",
            va='center',
            fontsize=8
        )

    # Add reference line at 0.5
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Subreddit factuality plot saved to: {output_path}")

    # Also save subreddit statistics to CSV
    csv_path = output_path.replace('.png', '_stats.csv')
    subreddit_stats.to_csv(csv_path, index=False)
    print(f"Subreddit statistics saved to: {csv_path}")


def plot_model_comparison(
    metrics_dict: Dict[str, dict],
    output_path: str,
    figsize: tuple = (10, 6)
):
    """
    Create bar plot comparing multiple models.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        output_path: Path to save the plot
        figsize: Figure size
    """
    # Extract metrics
    model_names = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in model_names]
    macro_f1s = [metrics_dict[m]['macro_f1'] for m in model_names]

    # Get ROC-AUC if available
    roc_aucs = [metrics_dict[m].get('roc_auc', None) for m in model_names]
    has_roc = any(r is not None for r in roc_aucs)

    # Create plot
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, macro_f1s, width, label='Macro F1', alpha=0.8, edgecolor='black')

    if has_roc:
        roc_aucs_clean = [r if r is not None else 0 for r in roc_aucs]
        bars3 = ax.bar(x + width, roc_aucs_clean, width, label='ROC-AUC', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.01,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

    add_labels(bars1)
    add_labels(bars2)
    if has_roc:
        add_labels(bars3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Model comparison plot saved to: {output_path}")


def plot_feature_importance(
    features_df: pd.DataFrame,
    output_path: str,
    top_n: int = 30,
    figsize: tuple = (10, 12)
):
    """
    Plot top feature importances from logistic regression.

    Args:
        features_df: DataFrame with 'feature', 'coefficient', 'direction' columns
        output_path: Path to save the plot
        top_n: Number of top features to display
        figsize: Figure size
    """
    # Get top N by absolute coefficient
    top_features = features_df.nlargest(top_n, 'abs_coefficient')

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['green' if d == 'positive' else 'red' for d in top_features['direction']]

    bars = ax.barh(
        range(len(top_features)),
        top_features['coefficient'],
        color=colors,
        alpha=0.7,
        edgecolor='black'
    )

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=9)
    ax.set_xlabel('Coefficient', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances\n(Green=Factual, Red=Non-Factual)',
                 fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Feature importance plot saved to: {output_path}")


def plot_error_analysis(
    df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (12, 5)
):
    """
    Visualize error patterns.

    Args:
        df: DataFrame with predictions and true labels
        output_path: Path to save the plot
        figsize: Figure size
    """
    if 'factuality_label' not in df.columns:
        print("Warning: True labels not available. Skipping error analysis plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Error rate by confidence
    df_copy = df.copy()
    df_copy['correct'] = (df_copy['pred_label'] == df_copy['factuality_label']).astype(int)
    df_copy['confidence'] = df_copy[['prob_nonfactual', 'prob_factual']].max(axis=1)
    df_copy['confidence_bin'] = pd.cut(df_copy['confidence'], bins=10)

    error_by_conf = df_copy.groupby('confidence_bin')['correct'].agg(['mean', 'count']).reset_index()
    error_by_conf['error_rate'] = 1 - error_by_conf['mean']

    # Get bin centers for x-axis
    bin_centers = [interval.mid for interval in error_by_conf['confidence_bin']]

    ax1.plot(bin_centers, error_by_conf['error_rate'], marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Prediction Confidence', fontsize=12)
    ax1.set_ylabel('Error Rate', fontsize=12)
    ax1.set_title('Error Rate by Prediction Confidence', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(error_by_conf['error_rate']) * 1.2])

    # Plot 2: Confusion by confidence
    high_conf = df_copy[df_copy['confidence'] > 0.8]
    low_conf = df_copy[df_copy['confidence'] <= 0.8]

    conf_levels = ['High Confidence\n(>0.8)', 'Low Confidence\n(≤0.8)']
    error_rates = [
        1 - high_conf['correct'].mean() if len(high_conf) > 0 else 0,
        1 - low_conf['correct'].mean() if len(low_conf) > 0 else 0
    ]
    counts = [len(high_conf), len(low_conf)]

    bars = ax2.bar(conf_levels, error_rates, alpha=0.7, edgecolor='black', color=['skyblue', 'salmon'])

    ax2.set_ylabel('Error Rate', fontsize=12)
    ax2.set_title('Error Rate by Confidence Level', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, max(error_rates) * 1.3 if max(error_rates) > 0 else 1])

    # Add count annotations
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'n={count}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Error analysis plot saved to: {output_path}")


def create_comprehensive_report(
    logreg_metrics_path: str,
    distilbert_metrics_path: str,
    alignment_results_path: str,
    output_dir: str
):
    """
    Create a comprehensive visualization report combining all analyses.

    Args:
        logreg_metrics_path: Path to logistic regression metrics JSON
        distilbert_metrics_path: Path to DistilBERT metrics JSON
        alignment_results_path: Path to alignment results CSV
        output_dir: Directory to save visualizations
    """
    import json

    print("\nCreating comprehensive visualization report...")

    # Load metrics
    with open(logreg_metrics_path, 'r') as f:
        logreg_metrics = json.load(f)

    with open(distilbert_metrics_path, 'r') as f:
        distilbert_metrics = json.load(f)

    # Model comparison
    metrics_dict = {
        'Logistic Regression': logreg_metrics,
        'DistilBERT': distilbert_metrics
    }

    comparison_path = os.path.join(output_dir, 'model_comparison.png')
    plot_model_comparison(metrics_dict, comparison_path)

    # Load alignment results if available
    if os.path.exists(alignment_results_path):
        alignment_df = pd.read_csv(alignment_results_path)

        # Similarity distributions
        sim_dist_path = os.path.join(output_dir, 'similarity_distribution_factual_vs_nonfactual.png')
        plot_similarity_distributions(alignment_df, sim_dist_path)

        # Probability vs similarity scatter
        scatter_path = os.path.join(output_dir, 'prob_vs_similarity_scatter.png')
        plot_probability_vs_similarity_scatter(alignment_df, scatter_path)

        # Subreddit analysis
        subreddit_path = os.path.join(output_dir, 'subreddit_factuality_barplot.png')
        plot_subreddit_factuality(alignment_df, subreddit_path)

        # Error analysis
        error_path = os.path.join(output_dir, 'error_analysis.png')
        plot_error_analysis(alignment_df, error_path)

    print("\nComprehensive report created successfully!")


if __name__ == '__main__':
    # This module is meant to be imported, but can also be run standalone
    # with specific arguments for generating visualizations
    print("Visualization utilities module. Import and use functions as needed.")
