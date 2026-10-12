# Hausa Sentiment Analysis

The sentiment analysis project uses Hausa-language social-media text from the
Masakhane AfriSenti Hausa dataset. We experimented with various approaches, including classical
machine-learning and neural-network approaches, such as TF-IDF word and
character features, recurrent models, fastText embeddings, and a multilingual
Transformer model.

- The complete experimental workflow is available in
[`notebook/hausa_sentiment_analysis.ipynb`](./notebook/hausa_sentiment_analysis.ipynb).
- The deployed app is available via Render at: [https://hausa-sentiment-analysis-app.onrender.com/](url)

## Repository structure

```text
.
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

## Requirements

- Python 3.12 (recommended)
- A Jupyter environment for running the notebook
- Optional: a CUDA-enabled installation of PyTorch or TensorFlow for
  hardware-accelerated experiments

The notebook uses the Hugging Face `datasets` and `transformers` libraries,
scikit-learn, TensorFlow, PyTorch, Gensim, and supporting data-visualization
packages. See [`requirements.txt`](./requirements.txt) for the full list.

## Setup

From the repository root, create and activate a virtual environment:

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

To run the notebook locally, install and start Jupyter if it is not already
available:

```bash
python -m pip install notebook
python -m notebook
```

Then open `notebook/hausa_sentiment_analysis.ipynb`.

## Dataset

The dataset files are CSVs with the following columns:

| Column | Description |
| --- | --- |
| `tweet` | Hausa social-media text |
| `label` | Sentiment label: `positive`, `neutral`, or `negative` |

The data is split into training, validation, and test sets:

- [`dataset/train.csv`](./dataset/train.csv)
- [`dataset/validation.csv`](./dataset/validation.csv)
- [`dataset/test.csv`](./dataset/test.csv)

The notebook can also load the Hausa (`hau`) configuration directly from the
[`masakhane/afrisenti`](https://huggingface.co/datasets/masakhane/afrisenti)
dataset on Hugging Face.

## Run the App

We are using Gradio web interface for this project. To launch it:

```bash
python src/app.py
```
