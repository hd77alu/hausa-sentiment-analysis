from functools import lru_cache
from pathlib import Path
import pickle

from src.preprocessing import clean_tweet


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "hybrid_tfidf_svm_model.pkl"
)

LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


@lru_cache(maxsize=1)
def load_model():
    """Load the trained classifier once and reuse it for later predictions."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}"
        )

    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def predict_hausa_sentiment(user_text: str) -> tuple[dict[str, float], str]:
    """Return class probabilities and the highest-probability sentiment."""
    cleaned_text = clean_tweet(user_text)
    model = load_model()

    probabilities = model.predict_proba([cleaned_text])[0]
    confidences = {
        LABELS[int(class_id)]: float(probability)
        for class_id, probability in zip(model.classes_, probabilities)
    }
    main_label = max(confidences, key=lambda label: confidences[label])

    return confidences, main_label
