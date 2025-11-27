"""
Training script for DistilBERT transformer classifier.

This script fine-tunes DistilBERT on Reddit posts from the FACTOID dataset
for factuality classification.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_factoid
from preprocess import apply_cleaning, train_val_test_split, remove_empty_texts
from model_distilbert import (
    create_distilbert_model,
    load_tokenizer,
    save_model,
    create_data_loader
)
from evaluate_models import (
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
    save_metrics_to_json,
    save_classification_report,
    plot_roc_curve,
    plot_training_curves
)
from utils import (
    save_predictions,
    set_random_seeds,
    create_output_dir,
    print_section,
    EarlyStopping
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train DistilBERT classifier for factuality classification'
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
        default='results/distilbert',
        help='Directory to save results and models'
    )

    # Model arguments
    parser.add_argument(
        '--model_name',
        type=str,
        default='distilbert-base-uncased',
        help='Pretrained DistilBERT model name'
    )

    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=256,
        help='Maximum sequence length for tokenization'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout probability'
    )

    # Training arguments
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Training batch size'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help='Learning rate'
    )

    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=0,
        help='Number of warmup steps for learning rate scheduler'
    )

    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Maximum gradient norm for clipping'
    )

    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=3,
        help='Early stopping patience (epochs without improvement)'
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
        '--num_workers',
        type=int,
        default=0,
        help='Number of workers for data loading'
    )

    return parser.parse_args()


def train_epoch(model, data_loader, optimizer, scheduler, device, max_grad_norm):
    """
    Train for one epoch.

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(data_loader, desc='Training')

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']

        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Update weights
        optimizer.step()
        scheduler.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, data_loader, device):
    """
    Evaluate model on validation/test set.

    Returns:
        Tuple of (avg_loss, predictions, true_labels, probabilities)
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    all_preds = []
    all_labels = []
    all_probs = []

    loss_fn = nn.CrossEntropyLoss()

    progress_bar = tqdm(data_loader, desc='Evaluating')

    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            # Compute loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches

    return avg_loss, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    """Main training pipeline."""
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
        print(f"\nUsing device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"\nUsing device: {device} (no GPU acceleration)")

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
    # STEP 2: Initialize tokenizer and model
    # =========================================================================
    print_section("STEP 2: Initializing Model and Tokenizer")

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_name)

    # Create model
    num_labels = len(df['factuality_label'].unique())
    model = create_distilbert_model(
        num_labels=num_labels,
        model_name=args.model_name,
        dropout=args.dropout
    )

    model.to(device)

    # =========================================================================
    # STEP 3: Create data loaders
    # =========================================================================
    print_section("STEP 3: Creating Data Loaders")

    train_loader = create_data_loader(
        texts=train_df['text_clean'].tolist(),
        labels=train_df['factuality_label'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = create_data_loader(
        texts=val_df['text_clean'].tolist(),
        labels=val_df['factuality_label'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_loader = create_data_loader(
        texts=test_df['text_clean'].tolist(),
        labels=test_df['factuality_label'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # =========================================================================
    # STEP 4: Setup training
    # =========================================================================
    print_section("STEP 4: Setting Up Training")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        mode='max'
    )

    print(f"\nTraining configuration:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {args.warmup_steps}")

    # =========================================================================
    # STEP 5: Training loop
    # =========================================================================
    print_section("STEP 5: Training Model")

    train_losses = []
    val_losses = []
    val_f1_scores = []
    best_val_f1 = 0.0

    for epoch in range(args.num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print('='*70)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.max_grad_norm
        )
        train_losses.append(train_loss)

        print(f"\nTrain Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, device)
        val_losses.append(val_loss)

        # Compute validation metrics
        class_names = ['Non-Factual', 'Factual']
        val_metrics = compute_metrics(val_labels, val_preds, val_probs, class_names)
        val_f1_scores.append(val_metrics['macro_f1'])

        print(f"\nValidation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")

        # Save best model
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_model_dir = os.path.join(args.output_dir, 'best_model')
            save_model(model, tokenizer, best_model_dir)
            print(f"\n✓ Best model saved (F1: {best_val_f1:.4f})")

        # Early stopping check
        if early_stopping(val_metrics['macro_f1']):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # Save final model
    final_model_dir = os.path.join(args.output_dir, 'final_model')
    save_model(model, tokenizer, final_model_dir)

    # =========================================================================
    # STEP 6: Evaluate on test set
    # =========================================================================
    print_section("STEP 6: Test Set Evaluation")

    # Load best model for testing
    from model_distilbert import load_model as load_distilbert_model
    best_model_dir = os.path.join(args.output_dir, 'best_model')
    model, _ = load_distilbert_model(best_model_dir, device)

    # Evaluate on test set
    test_loss, test_preds, test_labels, test_probs = evaluate(model, test_loader, device)

    test_metrics = compute_metrics(test_labels, test_preds, test_probs, class_names)
    print_metrics(test_metrics, title="Test Set Metrics")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    save_metrics_to_json(test_metrics, metrics_path)

    # Save classification report
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    save_classification_report(test_labels, test_preds, class_names, report_path)

    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    cm = np.array(test_metrics['confusion_matrix'])
    plot_confusion_matrix(cm, class_names, cm_path, normalize=False)

    cm_norm_path = os.path.join(args.output_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(cm, class_names, cm_norm_path, normalize=True)

    # Plot ROC curve
    if 'roc_auc' in test_metrics:
        roc_path = os.path.join(args.output_dir, 'roc_curve.png')
        plot_roc_curve(test_labels, test_probs[:, 1], roc_path, class_name='Factual')

    # Save predictions
    pred_path = os.path.join(args.output_dir, 'test_predictions.csv')
    save_predictions(
        predictions=test_preds,
        probabilities=test_probs,
        output_path=pred_path,
        post_ids=test_df['post_id'].values if 'post_id' in test_df.columns else None,
        true_labels=test_labels
    )

    # =========================================================================
    # STEP 7: Plot training curves
    # =========================================================================
    print_section("STEP 7: Saving Training Curves")

    curves_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        val_metrics=val_f1_scores,
        metric_name='Macro F1',
        output_path=curves_path
    )

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
    print(f"  - Best Model: {best_model_dir}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Confusion Matrix: {cm_path}")
    print(f"  - Training Curves: {curves_path}")
    print(f"  - Predictions: {pred_path}")


if __name__ == '__main__':
    main()
