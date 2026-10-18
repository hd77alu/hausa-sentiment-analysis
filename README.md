# Hausa Sentiment Analysis

This sentiment analysis project uses Hausa-language social-media text from the
[Masakhane AfriSenti Dataset](https://huggingface.co/datasets/masakhane/afrisenti). 
We experimented with various approaches, including classical
machine-learning and neural-network approaches, such as TF-IDF word and
character features, recurrent models, fastText embeddings, and a multilingual
Transformer model.

- Complete experiments: [`notebook/hausa_sentiment_analysis.ipynb`](./notebook/hausa_sentiment_analysis.ipynb)
- Deployed application: [https://hausa-sentiment-analysis-app.onrender.com](https://hausa-sentiment-analysis-app.onrender.com)

## Problem Statement

Hausa is widely spoken across West Africa, but it has substantially fewer
labelled datasets and NLP resources than English and other high-resource
languages. Sentiment analysis is especially challenging because social-media
text contains informal spelling, slang, emojis, code-switching, user mentions,
URLs, and dialectal variation.

This project addresses the problem of automatically identifying the sentiment
expressed in Hausa social-media posts. A reliable classifier can support
opinion monitoring, customer feedback analysis, public-health communication,
brand analysis, and civic engagement. The work compares classical machine
learning, neural sequence models, and a pretrained African-language
Transformer before selecting a practical model for deployment.


## Repository structure

```text
├── dataset/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── model/
│   └── hybrid_tfidf_svm_model.pkl
├── notebook/
│   └── hausa_sentiment_analysis.ipynb
├── src/
│   ├── app.py
│   ├── prediction.py
│   └── preprocessing.py
├── requirements.txt
└── README.md
```
## Set-Up

From the repository root, create and activate a
virtual environment, then install the dependencies (Python 3.12 is recommended):

### Windows Git Bash

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the notebook

```bash
python -m pip install notebook
python -m notebook
```

Open [`notebook/hausa_sentiment_analysis.ipynb`](./notebook/hausa_sentiment_analysis.ipynb)
and run the cells in order. The neural and Transformer experiments may
benefit from a CUDA-enabled environment.

## Run the application

The application uses Gradio and the saved calibrated hybrid SVM:

```bash
python src/app.py
```

The interface accepts Hausa text and returns the predicted sentiment together
with class-probability estimates. Inference is implemented in
[`src/prediction.py`](./src/prediction.py).

## Dataset

The project uses the Hausa (`hau`) configuration of the
[Masakhane AfriSenti dataset](https://huggingface.co/datasets/masakhane/afrisenti).
Each record contains a Hausa tweet and one of three sentiment labels:

| Column | Description |
| --- | --- |
| `tweet` | Hausa social-media text |
| `label` | `negative`, `neutral`, or `positive` |

The released splits used in the notebook are:

| Split | Examples |
| --- | ---: |
| Training | 14,172 |
| Validation | 2,677 |
| Test | 5,303 |
| **Total** | **22,152** |

The training set is reasonably balanced: `neutral` has 4,912 examples
(34.66%), `positive` has 4,687 (33.07%), and `negative` has 4,573 (32.27%).
Training tweets contain an average of approximately 78 characters and 14
words, with substantial variation in length and spelling.

Local copies of the splits are available at:

- [`dataset/train.csv`](./dataset/train.csv)
- [`dataset/validation.csv`](./dataset/validation.csv)
- [`dataset/test.csv`](./dataset/test.csv)

## Notebook

The notebook follows a reproducible supervised-learning workflow:

1. Load and audit the AfriSenti Hausa split.
2. Explore label balance, tweet lengths, vocabulary, and confusion patterns.
3. Normalize each tweet with the shared preprocessing pipeline.
4. Train several model families using the training split.
5. Select hyperparameters and compare models on the validation split.
6. Evaluate the strongest classical and deep-learning candidates on the
   held-out test split.
7. Save the deployment candidate and serve predictions through Gradio.

### Text preprocessing

[`src/preprocessing.py`](./src/preprocessing.py) implements the deployed
cleaning function. It:

- removes HTML entities, Twitter handles, and URLs;
- normalizes Hausa/African apostrophe variants;
- removes emoji modifiers and zero-width joiners;
- preserves emojis and flags as separated tokens because they can carry
  sentiment information; and
- normalizes repeated whitespace.

The preprocessing is intentionally conservative: it removes social-media
noise without stripping Hausa characters or sentiment-bearing content.

## Models

The experiments compare the following approaches:

### Hybrid TF-IDF + SVM (deployed)

The main classical model combines:

- word TF-IDF features with word n-grams `(1, 2)`;
- character TF-IDF features with character n-grams `(3, 5)`; and
- a `LinearSVC` classifier with `C=0.5`.

The word features capture sentiment-bearing words and short phrases, while
character features are useful for Hausa morphology, spelling variation,
misspellings, and informal social-media text. The saved model wraps the SVM
with probability calibration so the application can return class
confidences:
[`model/hybrid_tfidf_svm_model.pkl`](./model/hybrid_tfidf_svm_model.pkl).

### Neural sequence models

The notebook also evaluates unidirectional and regularized LSTMs, a
bidirectional LSTM, fastText-initialized BiLSTM models, a word/character
hybrid BiLSTM, and a hybrid GRU. These experiments test whether learned
sequential and sub-word representations improve on sparse TF-IDF features.

### AfriBERTa

The Transformer experiment fine-tunes
[`castorini/afriberta_large`](https://huggingface.co/castorini/afriberta_large)
for three-way Hausa sentiment classification. Its pretrained contextual
representations provide the strongest validation performance in the notebook,
especially for context-dependent and neutral examples.

## Validation Experiments Results

Accuracy, weighted precision, weighted recall, weighted F1, and one-vs-rest
ROC-AUC are used for comparison. The following results are from the
validation-set comparison reported in the notebook:

| Model | Validation accuracy | Weighted F1 | OVR AUC |
| --- | ---: | ---: | ---: |
| TF-IDF + Linear SVM | 73.37% | 73.45% | 0.8908 |
| Hybrid TF-IDF (word + character) + SVM | **77.51%** | **77.62%** | **0.9181** |
| Baseline LSTM | 74.11% | 74.01% | 0.8905 |
| Regularized BiLSTM | 74.04% | 74.04% | 0.8907 |
| Fine-tuned fastText BiLSTM | 74.64% | 74.62% | 0.8941 |
| fastText BiLSTM hybrid (word + character) | 75.64% | 75.64% | 0.9043 |
| Hybrid GRU (word + character) | 74.30% | 74.30% | 0.8960 |
| AfriBERTa Transformer | **79.94%** | **79.94%** | **0.9279** |

## Test Set Results
The top two performing models, Hybrid TF-IDF (Word & Char) and AfriBERTa, were selected for the final evaluation on the test set.

| **Model** | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **AUC (OVR)** |
|-----------|--------------|---------------|------------|--------------|---------------|
| Hybrid Word/Char TF-IDF with SVM | 78.01% | 78.32% | 78.01% | 78.05% | 0.9158 |
| AfriBERTa Transformer | 78.58% | 78.71% | 78.58% | 78.54% | 0.9251 |

## Main Findings

The main findings are:

- Character n-grams substantially improve the classical model, particularly
  for the difficult neutral class and noisy spelling patterns.
- The fastText word/character BiLSTM is the strongest sequence-only model, but
  remains below the hybrid SVM on the reported validation run.
- AfriBERTa achieves the best validation score by using pretrained contextual
  representations, but it requires considerably more memory and compute.
- On the held-out test evaluation reported in the notebook, the deployed
  hybrid SVM achieves **78.01% accuracy** and **78.05% weighted F1**.
- The hybrid SVM was selected for deployment instead of AfriBERTa because it
  has a small model footprint, low CPU latency, simpler dependencies, and no
  GPU requirement. This makes it more suitable for the current lightweight
  Render deployment.
