# Hausa Sentiment Analysis

## Deployment branch

This is the `deployment` branch of the project. It contains the lightweight
Gradio application and the trained model required to serve Hausa sentiment
predictions online. Notebook files, datasets, and training-only dependencies
are excluded from this branch so that deployment remains smaller and faster to
install.

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
