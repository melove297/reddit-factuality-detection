# Detecting Factual Reliability in Reddit Posts with FACTOID and Reuters

*A machine learning research project comparing TF-IDF baseline with DistilBERT transformers for detecting factual reliability in Reddit posts, validated against professional news articles from Reuters.*

**Authors:** Ria Kapoor, Bitanya Kebede, Charlie King, George Wright<br>
**Institution:** Duke University<br>
**Course:** CS376

## Project Overview

This project investigates whether models trained on Reddit factuality labels can learn patterns of truthfulness and generalize beyond platform-specific biases. Using the FACTOID corpus as our main source of labeled Reddit posts, we:

1. Compare a **TF-IDF + Logistic Regression baseline** with a **DistilBERT transformer classifier**
2. Evaluate how these models align with an external, professionally curated reference corpus from **Reuters news articles**
3. Analyze disagreement patterns between Reddit-trained models and Reuters similarity

## Project Structure

```
Detecting Factual Reliability in Reddit Posts/
│
├── src/                          # Source code (12 Python modules, ~3,900 LOC)
│   ├── __init__.py               # Package initialization
│   ├── data_loader.py            # FACTOID and Reuters data loading
│   ├── preprocess.py             # Text cleaning and data splitting
│   ├── utils.py                  # Metadata normalization utilities
│   ├── features_tfidf.py         # TF-IDF feature extraction
│   ├── model_logreg.py           # Logistic regression wrapper
│   ├── model_distilbert.py       # DistilBERT classifier
│   ├── train_logreg.py           # Logistic regression training script
│   ├── train_distilbert.py       # DistilBERT training script
│   ├── reuters_alignment.py      # External validation analysis
│   ├── evaluate_models.py        # Model evaluation metrics
│   └── visualize_results.py      # Visualization utilities
│
├── data/                         # Datasets (849MB)
│   ├── factoid_clean.csv         # 3,193,625 Reddit posts with labels
│   └── reuters.csv               # 10,788 Reuters news articles
│
├── results/                      # Training results and models (26MB+)
│   ├── logreg_full/              # Complete baseline results
│   │   ├── logreg_model.pkl      # Trained model (391KB)
│   │   ├── tfidf_vectorizer.pkl  # TF-IDF vectorizer (1.7MB)
│   │   ├── test_predictions.csv  # Test predictions (24MB, 477,065 rows)
│   │   ├── test_metrics.json     # Performance metrics
│   │   ├── confusion_matrix.png  # Visualizations
│   │   └── top_features.csv      # Most influential n-grams
│   └── distilbert_full/          # DistilBERT results (in progress)
│
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ (with GPU support recommended)
- Transformers (Hugging Face)
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- tqdm, scipy

### Install Dependencies

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm scipy
```

## Datasets

### FACTOID Dataset

- **Size**: 840MB (3,193,625 posts after cleaning)
- **Source**: Reddit corpus with factuality labels
- **Labels**: Binary (0 = Non-Factual, 1 = Factual)
- **Distribution**: 60% Factual, 40% Non-Factual
- **Columns**: post_id, text, subreddit, upvotes, num_comments, factuality_label

### Reuters Dataset

- **Size**: 8.9MB (10,788 articles)
- **Source**: Reuters-21578 corpus via NLTK
- **Categories**: earnings, acquisitions, money-fx, grain, crude, trade, etc.
- **Columns**: article_id, title, body_text, category

## Usage

### 1. Train Logistic Regression Baseline

```bash
python -m src.train_logreg \
    --factoid_path data/factoid_clean.csv \
    --output_dir results/logreg \
    --max_features 50000 \
    --ngram_range 1 2 \
    --random_seed 42
```

**Key Arguments:**
- `--factoid_path`: Path to FACTOID CSV file
- `--output_dir`: Directory to save results
- `--max_features`: Maximum TF-IDF vocabulary size (default: 50000)
- `--ngram_range`: N-gram range (default: 1 2 for unigrams+bigrams)
- `--C`: Inverse regularization strength (default: 1.0)
- `--use_metadata`: Include metadata features (default: True)

### 2. Train DistilBERT Classifier

```bash
python -m src.train_distilbert \
    --factoid_path data/factoid_clean.csv \
    --output_dir results/distilbert \
    --num_epochs 3 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --max_seq_length 256
```

**Key Arguments:**
- `--factoid_path`: Path to FACTOID CSV file
- `--output_dir`: Directory to save results
- `--num_epochs`: Number of training epochs (default: 5)
- `--batch_size`: Training batch size (default: 16)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_seq_length`: Maximum token sequence length (default: 256)
- `--early_stopping_patience`: Early stopping patience (default: 3)

### 3. Run Reuters Alignment Analysis

```bash
python -m src.reuters_alignment \
    --factoid_path data/factoid_clean.csv \
    --reuters_path data/reuters.csv \
    --model_dir results/distilbert_full \
    --output_dir results/alignment \
    --top_k 5
```

**Key Arguments:**
- `--factoid_path`: Path to FACTOID CSV file
- `--reuters_path`: Path to Reuters CSV file
- `--model_dir`: Directory with trained DistilBERT model
- `--output_dir`: Directory to save alignment results
- `--top_k`: Number of top similar Reuters articles (default: 5)

## Results

### Logistic Regression Baseline

Trained on full FACTOID dataset (3.2M posts):

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 64.11% |
| **Macro F1** | 0.6268 |
| **Weighted F1** | 0.6411 |
| **ROC-AUC** | 0.6711 |
| **Training Time** | ~21 minutes (CPU) |

**Per-Class Performance:**
- **Non-Factual**: Precision 0.553, Recall 0.554, F1 0.554 (191,763 samples)
- **Factual**: Precision 0.700, Recall 0.700, F1 0.700 (285,302 samples)

**Top Predictive Features (TF-IDF):**
- Non-Factual indicators: "alarmists", "orange monster", "climate", "agw"
- Factual indicators: "novavax", "anti vax", "github new", "ms mcenany"

### DistilBERT Transformer

Trained on full FACTOID dataset (3.2M posts) for 1 epoch:

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 63.19% |
| **Macro F1** | 0.5409 |
| **Weighted F1** | 0.5810 |
| **ROC-AUC** | 0.6204 |
| **Training Time** | ~20 hours (Apple Silicon GPU) |

**Per-Class Performance:**
- **Non-Factual**: Precision 0.611, Recall 0.232, F1 0.336 (191,763 samples)
- **Factual**: Precision 0.636, Recall 0.901, F1 0.745 (285,302 samples)

**Analysis:**
- DistilBERT underperformed the LogReg baseline (63.19% vs 64.11%)
- Model shows strong bias toward predicting factual class (90% recall, 23% recall for non-factual)
- Likely underfitting due to only 1 epoch of training (3-5 epochs recommended)
- Each sample seen only 1× vs LogReg's ~15× effective iterations

### Reuters Alignment Analysis

External validation comparing DistilBERT predictions with Reuters news article similarity (50K sample):

**Similarity Metrics:**
- **Mean similarity to Reuters**: 0.713 (reasonable alignment with professional news)
- **Max similarity**: 0.9998 (most posts have highly similar Reuters articles)
- **Top-5 mean similarity**: 0.999

**Model Predictions Distribution:**
- **Non-Factual**: 8,227 posts (16.5%)
- **Factual**: 41,773 posts (83.5%)
- Heavy bias toward factual predictions

**Key Findings:**
- **Correlation**: -0.121 (weak negative correlation between factual predictions and Reuters similarity)
- **Statistical significance**: Highly significant differences between groups (p ≈ 0)
- **Disagreement cases**: 2.62% of posts predicted as high-confidence non-factual have high Reuters similarity
- Reddit posts show substantial semantic overlap with professional news content

## Output Files

### Logistic Regression (`results/logreg/`)

- `logreg_model.pkl`: Trained model
- `tfidf_vectorizer.pkl`: Fitted TF-IDF vectorizer
- `metadata_processor.pkl`: Metadata normalization processor
- `test_metrics.json`: Test set performance metrics
- `val_metrics.json`: Validation set metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `confusion_matrix_normalized.png`: Normalized confusion matrix
- `roc_curve.png`: ROC curve
- `top_features.csv`: Most important TF-IDF features
- `test_predictions.csv`: Predictions on test set
- `classification_report.txt`: Detailed classification report

### DistilBERT (`results/distilbert/`)

- `best_model/`: Best model checkpoint (by validation F1)
- `final_model/`: Final model after all epochs
- `test_metrics.json`: Test set performance metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `training_curves.png`: Loss and F1 curves over epochs
- `test_predictions.csv`: Predictions on test set
- `classification_report.txt`: Detailed classification report

### Reuters Alignment (`results/alignment/`)

- `alignment_results.csv`: Full alignment analysis results
- `alignment_summary.json`: Summary statistics
- `disagreement_examples.txt`: Examples of model-Reuters disagreements
- `similarity_distribution_factual_vs_nonfactual.png`: Similarity distributions
- `prob_vs_similarity_scatter.png`: Scatter plot of probability vs similarity
- `subreddit_factuality_barplot.png`: Factuality rates by subreddit

## Research Questions

### 1. Do BERT-based models outperform TF-IDF + Logistic Regression?

Compare metrics from both models:

```python
import json

with open('results/logreg/test_metrics.json') as f:
    logreg = json.load(f)

with open('results/distilbert/test_metrics.json') as f:
    bert = json.load(f)

print(f"LogReg - Acc: {logreg['accuracy']:.4f}, F1: {logreg['macro_f1']:.4f}")
print(f"BERT   - Acc: {bert['accuracy']:.4f}, F1: {bert['macro_f1']:.4f}")
```

### 2. Do factual predictions align with Reuters similarity?

Check alignment results:

```python
with open('results/alignment/alignment_summary.json') as f:
    summary = json.load(f)

print(f"Factual posts - Mean similarity: {summary['factual_mean_similarity']:.4f}")
print(f"Non-factual posts - Mean similarity: {summary['nonfactual_mean_similarity']:.4f}")
print(f"P-value: {summary['mann_whitney_u_pvalue']:.4e}")
```

### 3. How often does the model disagree with Reuters?

```python
print(f"High-conf factual + low Reuters sim: {summary['disagreement_case1_percent']:.2f}%")
print(f"High-conf non-factual + high Reuters sim: {summary['disagreement_case2_percent']:.2f}%")
```

### 4. Which subreddits have higher misinformation rates?

See `results/alignment/subreddit_factuality_barplot_stats.csv` for per-subreddit statistics.

## Code Quality Features

✅ **Production-ready code:**
- Type hints for function parameters
- Comprehensive docstrings (Google style)
- Error handling with informative messages
- Command-line interfaces with argparse
- Reproducible results with random seeds (seed=42)

✅ **Modular design:**
- Reusable components
- Clean separation of concerns
- Easy to extend and modify

✅ **Complete evaluation:**
- Multiple metrics (accuracy, F1, precision, recall, ROC-AUC)
- Confusion matrices (raw and normalized)
- Feature importance analysis
- Error analysis
- External validation against Reuters

## Technical Details

### Data Preprocessing
- Text cleaning: URLs, HTML, Reddit artifacts, non-ASCII removal
- Stratified split: 70% train, 15% validation, 15% test
- Empty text removal (0.41% of data)
- Metadata normalization: z-score for numeric, label encoding for categorical

### TF-IDF Features
- Vocabulary size: up to 50,000 features
- N-grams: unigrams + bigrams (1,2)
- Sparse matrix: ~99.96% sparsity
- Metadata concatenation: upvotes, comments, subreddit

### DistilBERT Configuration
- Base model: distilbert-base-uncased
- Classification head: Linear layer with dropout (0.1)
- Optimizer: AdamW with linear warmup
- Early stopping: validation F1 (patience=3)
- Gradient clipping: max_norm=1.0
- Device: Auto-detection (CUDA/MPS/CPU)

## Citation

If you use this code, please cite:

```bibtex
@article{kapoor2024factoid,
  title={Detecting Factual Reliability in Reddit Posts with FACTOID and Reuters},
  author={Kapoor, Ria and Kebede, Bitanya and King, Charlie and Wright, George},
  year={2024},
  institution={Duke University}
}
```

## References

- Sakketou et al. (2022). "FACTOID: A large-scale benchmark for factuality detection in social media"
- Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
- Sanh et al. (2019). "DistilBERT, a distilled version of BERT"

## License

This project is created for academic purposes at Duke University.

## Contact

For questions or issues, please contact:
- Ria Kapoor: ria.kapoor@duke.edu
- Bitanya Kebede: bitanya.kebede@duke.edu
- Charlie King: charlie.king@duke.edu
- George Wright: george.wright@duke.edu
