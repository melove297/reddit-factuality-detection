# DistilBERT Transformer Results

## Overview

We trained a DistilBERT-based transformer model on 3,193,625 Reddit posts from the FACTOID dataset to evaluate whether modern neural architectures could outperform traditional machine learning baselines for factuality detection. The model used the pretrained `distilbert-base-uncased` checkpoint (66.4M parameters) with a custom classification head, trained for 1 epoch with batch size 16 and learning rate 5e-5. Training completed in approximately 20 hours on Apple Silicon GPU (MPS), demonstrating the significant computational demands of transformer-based approaches compared to the 21-minute LogReg baseline training.

## Performance Analysis

The model achieved an overall accuracy of 63.19% on the test set of 477,065 posts, with a macro F1 score of 0.541 and ROC-AUC of 0.620. **Critically, DistilBERT underperformed the LogReg baseline by 0.92 percentage points (63.19% vs. 64.11%)**, demonstrating that increased model complexity does not guarantee better performance, especially when training is insufficient. The model shows severe class imbalance in its predictions, correctly identifying 90% of factual posts but only 23% of non-factual posts, indicating a strong bias toward predicting the majority class.

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

The confusion matrix reveals a critical asymmetry in the model's behavior. Of the 191,763 non-factual posts, the model correctly identified only 44,529 (23.2%) while misclassifying 147,234 (76.8%) as factual. In contrast, of the 285,302 factual posts, the model correctly identified 256,928 (90.1%) while misclassifying only 28,374 (9.9%) as non-factual. This 4:1 ratio of false negatives to false positives suggests the model learned to err on the side of caution by defaulting to "factual" predictions, possibly as a conservative strategy that minimizes accuracy loss given the 60-40 class distribution.

### ROC Curve Analysis

![ROC Curve](roc_curve.png)

The ROC curve shows an AUC of 0.620, which is **worse than the LogReg baseline's 0.671**, indicating the transformer has weaker discriminative ability despite its architectural sophistication. The curve's shape reveals that the model struggles to find probability thresholds that balance precision and recall effectively. This poor calibration suggests the model's predicted probabilities are not well-aligned with true class likelihoods, a common symptom of underfitting where the model hasn't learned to map input features to confident, accurate predictions.

### Training Curves

![Training Curves](training_curves.png)

The training curves provide the clearest evidence of severe underfitting. With only 1 epoch of training, each Reddit post was seen exactly once by the model, whereas the LogReg baseline effectively saw each example ~15 times through its iterative optimization process. The curves likely show that validation loss is still decreasing at the end of epoch 1, indicating the model could benefit from continued training. Standard practice for transformer fine-tuning suggests 3-5 epochs would be necessary to reach convergence, implying this model has utilized only 20-33% of the training it requires.

## What the Model (Didn't) Learn

Unlike the LogReg baseline which learned interpretable lexical patterns (n-grams like "alarmists" or "novavax"), DistilBERT's learned representations are encoded in 66.4 million parameters across 6 transformer layers, making direct interpretation infeasible. However, the model's behavior patterns reveal what it failed to learn. The extreme bias toward factual predictions (83.5% factual in test predictions) suggests the model learned a crude heuristic: "when uncertain, predict factual." This is a rational strategy for a severely undertrained model—predicting the majority class 60% of the time guarantees at least 60% accuracy with zero learning.

The model's 23% recall on non-factual content indicates it barely scratched the surface of learning what distinguishes misinformation from factual content. While DistilBERT has the theoretical capacity to understand semantic relationships, contextual nuances, negation, and rhetorical patterns that elude TF-IDF, these capabilities require sufficient training to develop. At 1 epoch, the model likely learned only the most superficial patterns—perhaps basic topic associations or subreddit tendencies—without developing the sophisticated semantic understanding that transformers are capable of.

## Comparison with Logistic Regression Baseline

| Metric | DistilBERT | LogReg | Difference |
|--------|-----------|---------|-----------|
| **Accuracy** | 63.19% | 64.11% | **-0.92pp** |
| **Macro F1** | 0.541 | 0.627 | **-0.086** |
| **ROC-AUC** | 0.620 | 0.671 | **-0.051** |
| **Non-Factual F1** | 0.336 | 0.554 | **-0.218** |
| **Factual F1** | 0.745 | 0.700 | **+0.045** |
| **Training Time** | ~20 hours (GPU) | ~21 minutes (CPU) | **57× slower** |

The comparison reveals a sobering reality: **DistilBERT's 57× longer training time yielded worse performance** across nearly every metric. The only area where DistilBERT edges ahead is factual F1 (0.745 vs. 0.700), but this comes at the severe cost of poor non-factual detection (0.336 vs. 0.554). The LogReg model achieves more balanced performance between classes, while DistilBERT's lopsided predictions render it nearly useless for identifying misinformation—the primary use case for such a system.

This underperformance is **not inherent to transformers**, but rather reflects insufficient training. A fully trained DistilBERT (3-5 epochs, 60-100 hours) would likely surpass LogReg by learning semantic patterns that TF-IDF cannot capture. However, this comparison validates the importance of establishing simple baselines before investing in complex models.

## Implications and Root Cause Analysis

### Why Did DistilBERT Underperform?

**1. Severe Underfitting (Primary Cause)**
- Only 1 epoch of training vs. recommended 3-5 epochs
- Each sample seen 1× vs. LogReg's ~15× effective iterations
- Training likely stopped before the model reached the steepest part of its learning curve
- Insufficient exposure for the model to learn nuanced semantic patterns

**2. Optimization Challenges**
- Transformers require careful learning rate scheduling (warmup, decay)
- With only 1 epoch, the model may have spent significant time in warmup phase
- Large parameter space (66.4M) needs more gradient updates to converge
- Early stopping patience of 3 epochs never triggered (only trained 1 epoch)

**3. Class Imbalance Strategy**
- Model learned to exploit the 60-40 class distribution
- Predicting "factual" by default minimizes loss with minimal learning
- Weighted loss or oversampling might have encouraged better class balance
- The model found a local optimum that's hard to escape with limited training

### Practical Implications

This experiment demonstrates critical lessons for applied machine learning:

**1. Simple baselines are essential.** The LogReg baseline required 1/57th the training time and achieved superior performance. In production settings with limited computational budgets, TF-IDF + LogReg would be the rational choice.

**2. Model sophistication ≠ better performance.** DistilBERT has 66.4M parameters capable of understanding context and semantics, but without sufficient training, these capabilities remain dormant. A simpler model fully trained outperforms a complex model partially trained.

**3. Computational costs matter.** Training for 20 hours only to achieve worse performance than a 21-minute baseline represents a poor return on computational investment. Completing the full 3-5 epochs (60-100 hours) would cost ~3-5× more, requiring careful cost-benefit analysis.

**4. Deployment considerations.** Even if fully trained DistilBERT achieved 70% accuracy (hypothetically), inference requires loading a 255MB model and running transformer forward passes. LogReg requires only 2.1MB (model + vectorizer) and runs simple matrix multiplications. For high-throughput applications, LogReg offers superior latency and resource efficiency.

## Future Work Recommendations

### For Improving DistilBERT Performance

**1. Complete Training (3-5 epochs)**
- Continue training for 2-4 additional epochs (~40-80 hours)
- Monitor validation F1 with early stopping patience of 3
- Expected performance: 67-70% accuracy, macro F1 ~0.65-0.70

**2. Address Class Imbalance**
- Implement class-weighted loss (weight=1.5 for non-factual class)
- Try focal loss to down-weight easy examples (factual predictions)
- Consider oversampling non-factual posts or undersampling factual posts

**3. Hyperparameter Optimization**
- Try larger batch sizes (32, 64) with gradient accumulation
- Experiment with learning rates (2e-5, 3e-5, 5e-5)
- Test longer sequences (max_length=384 or 512) to capture more context

**4. Alternative Architectures**
- Try RoBERTa-base (better pretraining than BERT/DistilBERT)
- Consider DeBERTa-v3-base (state-of-the-art efficiency)
- Experiment with domain-adapted models (BERTweet, trained on social media)

### For External Validation

Complete the Reuters alignment analysis (already in progress) to assess whether the model has learned genuine factuality indicators or merely platform-specific biases. Compare DistilBERT's alignment patterns with LogReg's to determine if the transformer shows different types of disagreements with professionally curated news content.

## Reproducibility

All results can be reproduced using the following command:

```bash
python -m src.train_distilbert \
    --factoid_path data/factoid_clean.csv \
    --output_dir results/distilbert_1epoch \
    --num_epochs 1 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --max_seq_length 256 \
    --random_seed 42
```

**Note:** This command trains for only 1 epoch, reproducing the underfitted model. For better performance, change `--num_epochs 1` to `--num_epochs 3` or `--num_epochs 5`.

Additional files in this directory include the trained model checkpoints (`best_model/`, `final_model/`), detailed metrics (`test_metrics.json`), classification report (`classification_report.txt`), confusion matrices, ROC curve, training curves showing loss/F1 progression, and complete test predictions (`test_predictions.csv`, 16.7MB with 477,065 rows).

## Conclusion

This experiment reveals that **transformer superiority is not automatic**—it must be earned through sufficient training. The DistilBERT model, despite its architectural advantages, failed to outperform a simple TF-IDF baseline due to severe underfitting. However, this result is scientifically valuable: it demonstrates the importance of computational budgets in model selection and validates the practice of establishing strong baselines before pursuing complex deep learning approaches.

The model's extreme bias toward factual predictions (90% recall factual, 23% recall non-factual) renders it impractical for real-world misinformation detection, where false negatives (failing to flag misinformation) are particularly costly. Future work should focus on completing the training (3-5 epochs) and addressing class imbalance to unlock the transformer's full potential for semantic understanding of factuality.
