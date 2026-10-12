# Hausa Sentiment Analysis

## Deployment branch

This is the `deployment` branch of the project. It contains the lightweight
Gradio application and the trained model required to serve Hausa sentiment
predictions online. Notebook files, datasets, and training-only dependencies
are excluded from this branch so that deployment remains smaller and faster to
install.we

## Repository structure

```text

├── model/
│   └── hybrid_tfidf_svm_model.pkl
│
├── src/
│   ├── app.py
│   ├── prediction.py
│   └── preprocessing.py
├── requirements.txt
└── README.md
```


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


## Run the App

Gradio is listed as a dependency and the web interface is implemented in
`src/app.py`. To launch it:

```bash
python src/app.py
```
