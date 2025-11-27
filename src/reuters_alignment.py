"""
Reuters alignment analysis for external validation.

This module implements semantic similarity analysis between Reddit posts
and Reuters news articles to evaluate whether factuality predictions align
with professionally curated news content.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind, mannwhitneyu

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_factoid, load_reuters
from preprocess import apply_cleaning
from model_distilbert import load_model, load_tokenizer, DistilBertDataset
from utils import set_random_seeds, create_output_dir, print_section


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Reuters alignment analysis for external validation'
    )

    # Data arguments
    parser.add_argument(
        '--factoid_path',
        type=str,
        required=True,
        help='Path to FACTOID dataset CSV file'
    )

    parser.add_argument(
        '--reuters_path',
        type=str,
        required=True,
        help='Path to Reuters dataset CSV file'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing trained DistilBERT model'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/alignment',
        help='Directory to save alignment analysis results'
    )

    # Processing arguments
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=256,
        help='Maximum sequence length for tokenization'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for encoding'
    )

    parser.add_argument(
        '--max_reuters_samples',
        type=int,
        default=10000,
        help='Maximum number of Reuters articles to use (for memory efficiency)'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of top similar Reuters articles to consider'
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    return parser.parse_args()


def compute_embeddings(
    texts: list,
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device
) -> np.ndarray:
    """
    Compute DistilBERT embeddings for a list of texts.

    Uses the [CLS] token representation from DistilBERT as the text embedding.

    Args:
        texts: List of text strings
        model: Trained DistilBERT model
        tokenizer: DistilBERT tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        device: Device to run on

    Returns:
        Numpy array of embeddings with shape (n_texts, hidden_dim)
    """
    model.eval()

    # Create dataset and dataloader
    dataset = DistilBertDataset(
        texts=texts,
        labels=None,  # No labels needed for embedding
        tokenizer=tokenizer,
        max_length=max_length
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    embeddings = []

    print(f"Computing embeddings for {len(texts)} texts...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Encoding'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get [CLS] token representation (hidden_state)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_embeddings = outputs['hidden_state'].cpu().numpy()

            embeddings.append(batch_embeddings)

    # Concatenate all batches
    embeddings = np.vstack(embeddings)

    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def compute_similarity_metrics(
    reddit_embeddings: np.ndarray,
    reuters_embeddings: np.ndarray,
    top_k: int = 5
) -> dict:
    """
    Compute similarity metrics between Reddit posts and Reuters articles.

    Args:
        reddit_embeddings: Reddit post embeddings (n_reddit, hidden_dim)
        reuters_embeddings: Reuters article embeddings (n_reuters, hidden_dim)
        top_k: Number of top similar articles to consider

    Returns:
        Dictionary with similarity metrics for each Reddit post
    """
    print(f"\nComputing cosine similarity...")
    print(f"  Reddit posts: {reddit_embeddings.shape[0]}")
    print(f"  Reuters articles: {reuters_embeddings.shape[0]}")

    # Compute pairwise cosine similarity
    # Shape: (n_reddit, n_reuters)
    similarity_matrix = cosine_similarity(reddit_embeddings, reuters_embeddings)

    print(f"  Similarity matrix shape: {similarity_matrix.shape}")

    # Compute metrics for each Reddit post
    metrics = {
        'max_similarity': np.max(similarity_matrix, axis=1),
        'mean_similarity': np.mean(similarity_matrix, axis=1),
        'top_k_mean_similarity': np.mean(
            np.sort(similarity_matrix, axis=1)[:, -top_k:], axis=1
        ),
        'median_similarity': np.median(similarity_matrix, axis=1),
    }

    # Also store top-k indices for analysis
    metrics['top_k_indices'] = np.argsort(similarity_matrix, axis=1)[:, -top_k:][:, ::-1]

    print(f"\nSimilarity metrics computed:")
    print(f"  Max similarity - Mean: {metrics['max_similarity'].mean():.4f}")
    print(f"  Mean similarity - Mean: {metrics['mean_similarity'].mean():.4f}")
    print(f"  Top-{top_k} mean similarity - Mean: {metrics['top_k_mean_similarity'].mean():.4f}")

    return metrics


def analyze_alignment(
    df: pd.DataFrame,
    similarity_metrics: dict,
    output_dir: str
):
    """
    Analyze alignment between factuality predictions and Reuters similarity.

    Args:
        df: DataFrame with Reddit posts and predictions
        similarity_metrics: Dictionary of similarity metrics
        output_dir: Directory to save analysis results
    """
    print_section("Alignment Analysis")

    # Add similarity metrics to DataFrame
    for metric_name, values in similarity_metrics.items():
        if metric_name != 'top_k_indices':
            df[metric_name] = values

    # =========================================================================
    # Analysis 1: Similarity by predicted label
    # =========================================================================
    print("\n1. Similarity Distribution by Predicted Label")
    print("-" * 70)

    for label in sorted(df['pred_label'].unique()):
        subset = df[df['pred_label'] == label]
        label_name = 'Factual' if label == 1 else 'Non-Factual'

        print(f"\n{label_name} (n={len(subset)}):")
        print(f"  Max similarity - Mean: {subset['max_similarity'].mean():.4f} ± {subset['max_similarity'].std():.4f}")
        print(f"  Mean similarity - Mean: {subset['mean_similarity'].mean():.4f} ± {subset['mean_similarity'].std():.4f}")
        print(f"  Top-k mean - Mean: {subset['top_k_mean_similarity'].mean():.4f} ± {subset['top_k_mean_similarity'].std():.4f}")

    # Statistical test
    factual_sim = df[df['pred_label'] == 1]['max_similarity']
    nonfactual_sim = df[df['pred_label'] == 0]['max_similarity']

    # T-test
    t_stat, t_pval = ttest_ind(factual_sim, nonfactual_sim)
    print(f"\nT-test (max_similarity):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {t_pval:.4e}")

    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = mannwhitneyu(factual_sim, nonfactual_sim, alternative='two-sided')
    print(f"\nMann-Whitney U test (max_similarity):")
    print(f"  U-statistic: {u_stat:.4f}")
    print(f"  p-value: {u_pval:.4e}")

    # =========================================================================
    # Analysis 2: Disagreement cases
    # =========================================================================
    print("\n2. Model-Reuters Disagreement Analysis")
    print("-" * 70)

    # Define thresholds
    high_conf_threshold = 0.8
    low_sim_threshold = 0.3
    high_sim_threshold = 0.5

    # Case 1: High confidence factual but low Reuters similarity
    case1 = df[
        (df['prob_factual'] >= high_conf_threshold) &
        (df['max_similarity'] < low_sim_threshold)
    ]

    print(f"\nCase 1: High-confidence FACTUAL but LOW Reuters similarity")
    print(f"  Count: {len(case1)} ({len(case1)/len(df)*100:.2f}%)")
    if len(case1) > 0:
        print(f"  Mean confidence: {case1['prob_factual'].mean():.4f}")
        print(f"  Mean max similarity: {case1['max_similarity'].mean():.4f}")

    # Case 2: High confidence non-factual but high Reuters similarity
    case2 = df[
        (df['prob_factual'] < (1 - high_conf_threshold)) &
        (df['max_similarity'] > high_sim_threshold)
    ]

    print(f"\nCase 2: High-confidence NON-FACTUAL but HIGH Reuters similarity")
    print(f"  Count: {len(case2)} ({len(case2)/len(df)*100:.2f}%)")
    if len(case2) > 0:
        print(f"  Mean confidence (non-factual): {(1 - case2['prob_factual']).mean():.4f}")
        print(f"  Mean max similarity: {case2['max_similarity'].mean():.4f}")

    # =========================================================================
    # Analysis 3: Correlation analysis
    # =========================================================================
    print("\n3. Correlation Analysis")
    print("-" * 70)

    correlation = df['prob_factual'].corr(df['max_similarity'])
    print(f"\nCorrelation (prob_factual vs max_similarity): {correlation:.4f}")

    # =========================================================================
    # Save results
    # =========================================================================

    # Save full results
    results_path = os.path.join(output_dir, 'alignment_results.csv')
    # Remove top_k_indices before saving (it's a nested array)
    df_save = df.drop(columns=['top_k_indices'], errors='ignore')
    df_save.to_csv(results_path, index=False)
    print(f"\nFull results saved to: {results_path}")

    # Save disagreement examples
    save_disagreement_examples(case1, case2, output_dir)

    # Save summary statistics
    summary = {
        'num_reddit_posts': len(df),
        'factual_mean_similarity': float(factual_sim.mean()),
        'factual_std_similarity': float(factual_sim.std()),
        'nonfactual_mean_similarity': float(nonfactual_sim.mean()),
        'nonfactual_std_similarity': float(nonfactual_sim.std()),
        't_test_statistic': float(t_stat),
        't_test_pvalue': float(t_pval),
        'mann_whitney_u_statistic': float(u_stat),
        'mann_whitney_u_pvalue': float(u_pval),
        'correlation_prob_similarity': float(correlation),
        'disagreement_case1_count': len(case1),
        'disagreement_case1_percent': float(len(case1) / len(df) * 100),
        'disagreement_case2_count': len(case2),
        'disagreement_case2_percent': float(len(case2) / len(df) * 100),
    }

    import json
    summary_path = os.path.join(output_dir, 'alignment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary statistics saved to: {summary_path}")


def save_disagreement_examples(case1_df, case2_df, output_dir, n_examples=5):
    """
    Save example posts from disagreement cases.

    Args:
        case1_df: DataFrame of case 1 (factual pred, low similarity)
        case2_df: DataFrame of case 2 (non-factual pred, high similarity)
        output_dir: Output directory
        n_examples: Number of examples to save
    """
    output_path = os.path.join(output_dir, 'disagreement_examples.txt')

    with open(output_path, 'w') as f:
        f.write("Reuters-Model Disagreement Examples\n")
        f.write("=" * 80 + "\n\n")

        # Case 1 examples
        f.write("CASE 1: High-confidence FACTUAL predictions with LOW Reuters similarity\n")
        f.write("-" * 80 + "\n\n")

        case1_sample = case1_df.nlargest(n_examples, 'prob_factual') if len(case1_df) > 0 else case1_df

        for idx, row in case1_sample.iterrows():
            f.write(f"Example {idx}:\n")
            f.write(f"  Predicted: Factual (confidence: {row['prob_factual']:.4f})\n")
            f.write(f"  Max Reuters similarity: {row['max_similarity']:.4f}\n")
            if 'true_label' in row:
                f.write(f"  True label: {row['true_label']}\n")
            text = str(row.get('text_clean', row.get('text', '')))[:500]
            f.write(f"  Text: {text}\n")
            f.write("\n" + "-" * 80 + "\n\n")

        # Case 2 examples
        f.write("\nCASE 2: High-confidence NON-FACTUAL predictions with HIGH Reuters similarity\n")
        f.write("-" * 80 + "\n\n")

        case2_sample = case2_df.nsmallest(n_examples, 'prob_factual') if len(case2_df) > 0 else case2_df

        for idx, row in case2_sample.iterrows():
            f.write(f"Example {idx}:\n")
            f.write(f"  Predicted: Non-Factual (confidence: {1 - row['prob_factual']:.4f})\n")
            f.write(f"  Max Reuters similarity: {row['max_similarity']:.4f}\n")
            if 'true_label' in row:
                f.write(f"  True label: {row['true_label']}\n")
            text = str(row.get('text_clean', row.get('text', '')))[:500]
            f.write(f"  Text: {text}\n")
            f.write("\n" + "-" * 80 + "\n\n")

    print(f"\nDisagreement examples saved to: {output_path}")


def main():
    """Main alignment analysis pipeline."""
    args = parse_args()

    # Set random seeds
    set_random_seeds(args.random_seed)

    # Create output directory
    create_output_dir(args.output_dir)

    # Determine device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\nUsing device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing device: {device} (NVIDIA GPU)")
    else:
        device = torch.device('cpu')
        print(f"\nUsing device: {device} (CPU)")

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    print_section("STEP 1: Loading Data")

    # Load FACTOID test set (or full dataset)
    print(f"\nLoading FACTOID dataset from: {args.factoid_path}")
    factoid_df = load_factoid(args.factoid_path)
    factoid_df = apply_cleaning(factoid_df, text_col='text')

    # Load Reuters articles
    print(f"\nLoading Reuters dataset from: {args.reuters_path}")
    reuters_df = load_reuters(args.reuters_path)
    reuters_df = apply_cleaning(reuters_df, text_col='text')

    # Limit Reuters samples for memory efficiency
    if len(reuters_df) > args.max_reuters_samples:
        print(f"\nSampling {args.max_reuters_samples} Reuters articles...")
        reuters_df = reuters_df.sample(n=args.max_reuters_samples, random_state=args.random_seed)

    # =========================================================================
    # STEP 2: Load model and get predictions
    # =========================================================================
    print_section("STEP 2: Loading Model and Getting Predictions")

    # Load model
    model, tokenizer = load_model(args.model_dir, device)

    # Get predictions for Reddit posts
    print("\nGetting predictions for Reddit posts...")

    from model_distilbert import create_data_loader

    reddit_loader = create_data_loader(
        texts=factoid_df['text_clean'].tolist(),
        labels=factoid_df['factuality_label'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Get predictions
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(reddit_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    factoid_df['pred_label'] = all_preds
    factoid_df['prob_nonfactual'] = [p[0] for p in all_probs]
    factoid_df['prob_factual'] = [p[1] for p in all_probs]

    # =========================================================================
    # STEP 3: Compute embeddings
    # =========================================================================
    print_section("STEP 3: Computing Embeddings")

    # Compute Reddit embeddings
    reddit_embeddings = compute_embeddings(
        texts=factoid_df['text_clean'].tolist(),
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=device
    )

    # Compute Reuters embeddings
    reuters_embeddings = compute_embeddings(
        texts=reuters_df['text'].tolist(),
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=device
    )

    # =========================================================================
    # STEP 4: Compute similarity metrics
    # =========================================================================
    print_section("STEP 4: Computing Similarity Metrics")

    similarity_metrics = compute_similarity_metrics(
        reddit_embeddings=reddit_embeddings,
        reuters_embeddings=reuters_embeddings,
        top_k=args.top_k
    )

    # Add similarity metrics to factoid_df
    for key, value in similarity_metrics.items():
        factoid_df[key] = value

    # =========================================================================
    # STEP 5: Analyze alignment
    # =========================================================================
    analyze_alignment(factoid_df, similarity_metrics, args.output_dir)

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Analysis Complete!")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
