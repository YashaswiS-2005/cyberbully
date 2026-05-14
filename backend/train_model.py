import os
import pickle

import nltk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from backend.text_preprocessing import preprocess_text, tokenize_text


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "cyberbullying_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pipeline.pkl")


def download_nltk_resources():
    """Download NLTK resources needed for preprocessing."""
    nltk.download("stopwords", quiet=True)

def build_pipeline() -> Pipeline:
    """Build a scikit-learn pipeline with TF-IDF and Logistic Regression."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=preprocess_text,
                tokenizer=tokenize_text,
                lowercase=False,
                ngram_range=(1, 2),
                max_df=0.95,
                min_df=1,
                max_features=20000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def train_and_save_model():
    """Train the model on the dataset and save the pipeline."""
    download_nltk_resources()
    data = pd.read_csv(DATA_PATH)
    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    X = data["text"].astype(str)
    y = data["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(pipeline, model_file)

    print(f"Saved trained model pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
