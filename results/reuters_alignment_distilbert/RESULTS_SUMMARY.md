# Reuters Alignment Analysis Results

## Overview

This analysis evaluates whether the DistilBERT model trained on Reddit factuality labels learns genuine patterns of truthfulness or merely platform-specific biases. We performed external validation by comparing the model's predictions on a stratified 50K sample of Reddit posts against semantic similarity to professionally curated Reuters news articles. The methodology uses DistilBERT embeddings to compute cosine similarity between each Reddit post and 10,788 Reuters articles, measuring whether posts predicted as "factual" show higher alignment with professional journalism than those predicted as "non-factual."

The analysis addresses a fundamental question in misinformation detection: **Can models trained on community-labeled social media data generalize beyond platform-specific patterns?** If the model simply learned to recognize writing styles, subreddit tendencies, or topic associations specific to Reddit's ecosystem, it would show poor alignment with external journalistic standards. Conversely, if the model learned transferable indicators of factuality, we would expect strong positive correlation between factual predictions and Reuters similarity.

## Methodology

### Data and Sampling

- **Reddit Sample**: 50,000 posts (stratified by factuality label from FACTOID test set)
- **Reuters Corpus**: 10,788 news articles from Reuters-21578 dataset
- **Embedding Model**: DistilBERT-base-uncased (same architecture used for classification)
- **Similarity Metric**: Cosine similarity between [CLS] token embeddings
- **Computation**: All 50K × 10.7K = 540M pairwise similarities computed in ~15 minutes

### Similarity Metrics Extracted

For each Reddit post, we computed:
- **max_similarity**: Highest similarity to any Reuters article (measures best-case alignment)
- **mean_similarity**: Average similarity across all 10,788 Reuters articles (measures overall alignment)
- **top_k_mean_similarity**: Average similarity to top-5 most similar articles (measures alignment with closest matches)
- **median_similarity**: Median similarity across all Reuters articles (robust to outliers)

## Key Findings

### 1. Overall Reddit-Reuters Semantic Alignment

**Mean Similarity to Reuters: 0.713**

Reddit posts show substantial semantic overlap with professional news content, with an average cosine similarity of 0.71 across all Reuters articles. This indicates that the topics, vocabulary, and discourse patterns in Reddit discussions are reasonably well-aligned with professional journalism, at least at a semantic level. The mean similarity of 0.71 suggests that Reddit posts are neither completely disconnected from news content (which would yield similarities near 0) nor identical to it (which would yield similarities near 1), but occupy a middle ground where they discuss similar topics in somewhat similar ways.

**Distribution of Similarities:**
- Mean similarity across all articles: **0.713**
- Maximum similarity (best Reuters match): **0.9998** (nearly perfect matches exist)
- Top-5 mean similarity: **0.999** (most posts have multiple highly similar Reuters articles)
- Standard deviation: ~0.02 (relatively consistent across posts)

The high max and top-5 similarities reveal that **for nearly every Reddit post, there exists at least one Reuters article that discusses highly similar content**. This validates the Reuters corpus as a reasonable reference standard for factuality validation—if Reddit posts were discussing entirely different topics than Reuters covers, this methodology would be invalid.

### 2. Model Prediction Distribution

**Severe Bias Toward Factual Class:**
- **Non-Factual predictions**: 8,227 posts (16.5%)
- **Factual predictions**: 41,773 posts (83.5%)

The model's prediction distribution reveals the same class imbalance observed in the full test set evaluation. Despite the stratified sample containing 50% non-factual posts, the underfitted DistilBERT model predicted only 16.5% as non-factual. This 3:1 false negative ratio demonstrates the model's conservative strategy of defaulting to "factual" when uncertain—a direct consequence of insufficient training (1 epoch vs. needed 3-5 epochs).

### 3. Correlation Between Predictions and Reuters Similarity

**Correlation Coefficient: -0.121 (Weak Negative)**

Counterintuitively, posts predicted as "factual" show **slightly lower** similarity to Reuters articles than posts predicted as "non-factual." This weak negative correlation suggests the model is **not** using semantic alignment with professional journalism as a basis for its factuality judgments. Instead, the model appears to be relying on other factors—likely superficial patterns like writing style, subreddit identity, or topic-specific associations learned from Reddit's internal labeling patterns.

**Statistical Significance:**
- Mann-Whitney U test p-value: ≈ 0 (highly significant)
- T-test p-value: 3.3 × 10⁻²⁶ (highly significant)

While the correlation is weak, it is highly statistically significant due to the large sample size (50K posts). The negative direction indicates a systematic pattern: the model's conception of "factuality" is **inversely related** to professional journalism standards, albeit weakly. This is a concerning finding that suggests the model learned platform-specific patterns rather than generalizable indicators of truthfulness.

### 4. Disagreement Cases Analysis

**High-Confidence Non-Factual + High Reuters Similarity: 2.62%**

A total of 1,312 posts (2.62% of the sample) were predicted as non-factual with high confidence (probability > 0.8) despite having high similarity to Reuters articles (max similarity > 0.99). These represent cases where the model's Reddit-learned patterns directly contradict professional journalism alignment. Examining these disagreement cases reveals important insights:

#### Example Disagreements

**Example 1: Factual News Headline Flagged as Misinformation**
- Text: *"postal worker admits fabricating allegations of ballot tampering, officials say"*
- Prediction: Non-Factual (confidence: 0.809)
- Max Reuters similarity: 1.000
- **Analysis**: This is a factual news headline that should be labeled factual, but the model flagged it as non-factual. The topic (ballot tampering allegations) may be associated with misinformation in the training data, causing the model to over-generalize.

**Example 2: Amazon Warehouse News**
- Text: *"exclusive: amazon deploys thermal cameras at warehouses to scan for fevers faster"*
- Prediction: Non-Factual (confidence: 0.809)
- Max Reuters similarity: 1.000
- **Analysis**: A straightforward factual statement about corporate pandemic response, but flagged as non-factual. The model may have learned to associate Amazon-related posts with non-factual content based on Reddit community biases.

**Example 3: Russian Invasion Sarcasm**
- Text: *"cnn breaking news: russia uses c19 health catastrophe in croatia to invade the country and recreate the sfr yugoslavia!"*
- Prediction: Non-Factual (confidence: 0.809)
- Max Reuters similarity: 1.000
- **Analysis**: This is clearly sarcastic/satirical content that is genuinely non-factual. The model correctly identified it as non-factual, but the high Reuters similarity reflects that it discusses real geopolitical topics (Russia, COVID-19) that are covered extensively in news.

**Example 4: Nuanced Discussion**
- Text: *"i guess i don t know the intention, but i know that it removed some antivax comments and allows others in a seemingly arbitrary manner..."*
- Prediction: Non-Factual (confidence: 0.809)
- Max Reuters similarity: 1.000
- **Analysis**: This is a subjective opinion/observation rather than a factual claim. The model may be correctly identifying it as non-factual (opinion-based) even though it discusses topics covered in news (vaccine discourse, content moderation).

### Interpretation of Disagreement Cases

The disagreement examples reveal **three distinct phenomena**:

1. **Model Over-Generalization**: Flagging factual news headlines as non-factual based on topic associations (Examples 1, 2)
2. **Legitimate Disagreements**: Correctly identifying satire/sarcasm even when semantically similar to news (Example 3)
3. **Opinion vs. Fact Distinction**: Identifying subjective statements as non-factual regardless of topic (Example 4)

Only the first category represents true model failures. The second and third categories suggest the model may be learning some valid distinctions that go beyond simple semantic similarity—sarcasm detection and opinion identification are legitimate aspects of factuality assessment that TF-IDF baselines cannot capture.

## Implications and Insights

### What This Analysis Reveals About the Model

**1. The Model Did Not Learn "Alignment with Professional Journalism"**

The weak negative correlation (-0.121) demonstrates that DistilBERT's factuality judgments are **not based on semantic similarity to Reuters content**. Instead, the model learned internal Reddit patterns—subreddit communities, writing styles, topic associations—that may not transfer to external contexts. This is unsurprising given that the model was trained exclusively on Reddit data with Reddit-internal labels and had only 1 epoch of exposure.

**2. Severe Underfitting Limits Generalization**

With 83.5% factual predictions, the model barely discriminates between classes, suggesting it hasn't developed sophisticated factuality representations. A fully trained model (3-5 epochs) might show different alignment patterns if it learned more nuanced semantic features rather than superficial topic associations.

**3. Reuters Similarity May Not Be the Gold Standard**

The disagreement examples reveal that **high Reuters similarity doesn't always indicate factuality**. Sarcastic content, opinions, and speculative discussions can be semantically similar to news while remaining non-factual. Conversely, some factual Reddit content (personal experiences, niche topics) may have low Reuters similarity simply because Reuters doesn't cover those topics. The correlation analysis assumes Reuters similarity = factuality, but this equivalence is imperfect.

**4. Platform-Specific Biases Are Present**

The model learned to associate certain topics (Amazon, ballot tampering, vaccines) with non-factuality based on Reddit community patterns, even when the content aligns with news. These topic-level associations represent platform-specific biases that would hurt generalization to other contexts.

### Comparison with Logistic Regression Expectations

While we don't have Reuters alignment results for the LogReg baseline, we can infer expected differences:

**LogReg Predictions Would Likely Show:**
- **Higher correlation with Reuters similarity**: TF-IDF features (word overlap) naturally correlate with semantic similarity, so posts with news-like vocabulary would score as more factual
- **Less sophisticated disagreements**: LogReg cannot detect sarcasm, opinions, or rhetorical nuance—it relies purely on lexical patterns
- **Different topic biases**: LogReg learned associations like "alarmists" → non-factual and "novavax" → factual, which are equally platform-specific but more interpretable

**DistilBERT's Advantages (If Fully Trained):**
- Could detect sarcasm, negation, and rhetorical patterns (as shown in Example 3)
- Could distinguish opinions from factual claims (Example 4)
- Could learn topic-agnostic features like argument structure, evidence citation, hedging language

The current underfitted DistilBERT shows hints of these capabilities (Examples 3-4) but hasn't developed them reliably.

## Limitations of This Analysis

### 1. Reuters as a Proxy for Factuality

**Assumption**: High semantic similarity to Reuters indicates factuality.

**Issues**:
- Reuters covers mainstream news; Reddit discusses personal experiences, niche topics, emerging events
- Satirical content about real events shows high similarity despite being non-factual (Example 3)
- Factual Reddit content about non-newsworthy topics shows low similarity
- Reuters from 1987 may not cover modern topics well (COVID-19, contemporary politics)

### 2. Semantic Similarity vs. Factual Agreement

Embeddings capture **topical similarity**, not **truth value**. Two texts can discuss the same topic (high similarity) while making opposite claims:
- "Vaccines are effective" and "Vaccines are ineffective" → high embedding similarity, opposite factuality
- DistilBERT embeddings encode semantics, not fact-checking logic

### 3. Sample Size and Coverage

With 50K posts and 10.7K Reuters articles, we cover a reasonable range of topics, but:
- Some Reddit communities may be underrepresented
- Reuters-21578 categories (earnings, acquisitions, grain) may not align with Reddit topics (politics, COVID-19, social issues)
- Stratified sampling ensured label balance but not topic balance

### 4. Model Underfitting Confounds Results

With only 1 epoch of training, the model hasn't developed mature representations. The weak negative correlation may reflect:
- Immature learned representations (noise, not signal)
- Over-reliance on superficial patterns due to insufficient training
- Different results might emerge with a fully trained model

## Future Directions

### 1. Run Alignment Analysis for LogReg Baseline

Compare LogReg's alignment patterns with DistilBERT's to determine whether:
- LogReg shows higher correlation (due to lexical overlap with news style)
- The two models disagree on different types of posts
- Either model shows systematically better alignment

### 2. Complete DistilBERT Training and Re-evaluate

Train DistilBERT for 3-5 epochs and repeat this analysis to assess whether:
- A fully trained model shows positive correlation with Reuters similarity
- The class imbalance reduces (less bias toward factual predictions)
- Disagreement cases shift from over-generalization to more nuanced distinctions

### 3. Enhanced Alignment Metrics

Develop metrics that go beyond simple semantic similarity:
- **Claim-level alignment**: Extract factual claims and verify against Reuters claims
- **Evidence patterns**: Measure presence of citations, hedging, epistemic markers
- **Rhetorical analysis**: Detect sarcasm, emotional language, argumentative patterns
- **Topic-adjusted similarity**: Normalize similarity within topic clusters

### 4. Multi-Source Validation

Expand beyond Reuters to include:
- Modern news sources (2020+ articles covering COVID-19, contemporary politics)
- Fact-checking databases (Snopes, PolitiFact, FactCheck.org)
- Academic sources (PubMed, arXiv for scientific claims)
- Wikipedia as a neutral reference corpus

## Reproducibility

All results can be reproduced using the following command:

```bash
python -m src.reuters_alignment \
    --factoid_path data/factoid_sample_50k.csv \
    --reuters_path data/reuters.csv \
    --model_dir results/distilbert_1epoch/best_model \
    --output_dir results/reuters_alignment_distilbert \
    --batch_size 64 \
    --top_k 5 \
    --random_seed 42
```

**To create the 50K stratified sample:**

```python
import pandas as pd

df = pd.read_csv('data/factoid_clean.csv')
sample = df.groupby('factuality_label', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 25000), random_state=42)
).sample(n=50000, random_state=42)
sample.to_csv('data/factoid_sample_50k.csv', index=False)
```

Additional files in this directory include the complete alignment results (`alignment_results.csv`, 27MB with 50,000 rows containing all similarity metrics per post), summary statistics (`alignment_summary.json`), and disagreement examples (`disagreement_examples.txt` with 5 example cases).

## Conclusion

This external validation reveals a sobering finding: **the DistilBERT model's factuality judgments show weak negative correlation with professional journalism similarity** (-0.121), indicating it learned platform-specific patterns rather than generalizable truthfulness indicators. The model's severe bias toward factual predictions (83.5%) and the presence of contradictory disagreement cases (factual news headlines flagged as misinformation) suggest insufficient training prevented the development of mature semantic understanding.

However, this result is scientifically valuable for several reasons:

**1. Validates the Need for External Validation**: Internal metrics (accuracy, F1) can be misleading. A model achieving 63% accuracy may still fail to learn generalizable patterns. External validation against reference corpora like Reuters is essential for assessing real-world applicability.

**2. Reveals Platform Bias Risks**: Models trained on platform-specific labels (Reddit votes/annotations) may learn community biases, subreddit tendencies, and topic associations that don't reflect objective factuality. Deployment to other contexts (Twitter, news comments, public discourse) would require careful evaluation.

**3. Highlights the Semantic Similarity Limitation**: High embedding similarity ≠ factuality. The disagreement examples show that sarcasm, opinions, and speculative content can be semantically similar to news while remaining non-factual. Future work needs richer validation methods beyond cosine similarity.

**4. Motivates Further Training**: The current analysis reflects a severely underfitted model (1 epoch). Repeating this analysis with a fully trained DistilBERT (3-5 epochs) would reveal whether proper training enables learning of generalizable factuality indicators or whether platform biases are inherent to Reddit-labeled training data.

The ultimate question remains unanswered: **Can supervised learning on Reddit labels teach models about factuality, or only about Reddit?** A fully trained model and comparison with the LogReg baseline will help answer this question in future work.
