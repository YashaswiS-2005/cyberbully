"""
Enhanced Model Training for Large Datasets
Supports incremental learning, distributed training, and data augmentation.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils import resample

import nltk
from backend.text_preprocessing import preprocess_text, tokenize_text

logger = logging.getLogger(__name__)

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "cyberbullying_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pipeline.pkl")
LARGE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "collected_data.csv")


def download_nltk_resources():
    """Download NLTK resources needed for preprocessing."""
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
    except Exception as e:
        logger.warning(f"NLTK download warning: {e}")


class LargeDatasetTrainer:
    """Trainer optimized for large datasets."""

    def __init__(
        self,
        vectorizer_params: Optional[Dict] = None,
        classifier_params: Optional[Dict] = None
    ):
        self.vectorizer_params = vectorizer_params or {
            "lowercase": False,
            "ngram_range": (1, 2),
            "max_features": 50000,
            "min_df": 1,
            "max_df": 0.95,
        }
        self.classifier_params = classifier_params or {
            "max_iter": 1000,
            "tol": 1e-4,
            "random_state": 42,
            "class_weight": "balanced",
        }
        self.pipeline = None
        self.training_stats = {}

    def build_pipeline(self, use_sgd: bool = False) -> Pipeline:
        """Build the ML pipeline."""
        vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text,
            tokenizer=tokenize_text,
            **self.vectorizer_params
        )
        
        if use_sgd:
            # SGDClassifier is better for large datasets
            classifier = SGDClassifier(
                loss="log_loss",  # logistic regression
                penalty="l2",
                alpha=1e-4,
                **self.classifier_params
            )
        else:
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                **self.classifier_params
            )

        return Pipeline([
            ("tfidf", vectorizer),
            ("classifier", classifier)
        ])

    def load_dataset(self, path: str) -> Tuple[pd.DataFrame, Dict]:
        """Load and analyze dataset."""
        logger.info(f"Loading dataset from: {path}")
        
        # Detect file size and choose loading strategy
        file_size = os.path.getsize(path) / (1024 * 1024)  # MB
        
        if file_size > 100:
            # Large file - load in chunks
            chunks = []
            for chunk in pd.read_csv(path, chunksize=50000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path)

        # Analyze dataset
        stats = {
            "total_samples": len(df),
            "file_size_mb": round(file_size, 2),
            "columns": list(df.columns),
            "label_distribution": df["label"].value_counts().to_dict() if "label" in df.columns else {}
        }
        
        logger.info(f"Loaded {stats['total_samples']} samples")
        return df, stats

    def balance_dataset(self, df: pd.DataFrame, strategy: str = "oversample") -> pd.DataFrame:
        """Balance dataset for better model performance."""
        if "label" not in df.columns:
            return df

        # Separate by class
        classes = df["label"].unique()
        dfs = [df[df["label"] == c] for c in classes]
        max_size = max(len(d) for d in dfs)
        min_size = min(len(d) for d in dfs)

        # Only balance if there's significant imbalance
        if max_size / min_size < 1.5:
            return df

        balanced_dfs = []
        for class_df in dfs:
            if strategy == "oversample":
                # Oversample minority class
                balanced = resample(
                    class_df,
                    replace=True,
                    n_samples=max_size,
                    random_state=42
                )
            else:
                # Undersample majority class
                balanced = resample(
                    class_df,
                    replace=False,
                    n_samples=min_size,
                    random_state=42
                )
            balanced_dfs.append(balanced)

        result = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Balanced dataset: {len(result)} samples")
        return result

    def train(
        self,
        data_path: str,
        test_size: float = 0.2,
        use_sgd: bool = True,
        balance: bool = True,
        cross_validate: bool = True
    ) -> Dict[str, Any]:
        """Train the model on the dataset."""
        download_nltk_resources()
        
        # Load data
        df, stats = self.load_dataset(data_path)
        
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")

        # Balance if requested
        if balance:
            df = self.balance_dataset(df)

        # Prepare features and labels
        X = df["text"].astype(str)
        y = df["label"].astype(str)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

        # Build and train pipeline
        self.pipeline = self.build_pipeline(use_sgd=use_sgd)
        
        logger.info("Starting training...")
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Cross-validation if requested
        cv_scores = []
        if cross_validate and len(X_train) > 1000:
            cv_scores = cross_val_score(
                self.pipeline, X_train, y_train, cv=5, scoring="f1_weighted"
            )

        # Store training stats
        self.training_stats = {
            "accuracy": accuracy,
            "f1_score": f1,
            "cv_scores": cv_scores.tolist() if len(cv_scores) > 0 else [],
            "train_size": len(X_train),
            "test_size": len(X_test),
            "trained_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Training complete. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return self.training_stats

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model."""
        save_path = path or MODEL_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump(self.pipeline, f)
        
        logger.info(f"Saved model to: {save_path}")
        return save_path

    def load_model(self, path: Optional[str] = None) -> Pipeline:
        """Load a trained model."""
        load_path = path or MODEL_PATH
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        with open(load_path, "rb") as f:
            self.pipeline = pickle.load(f)
        
        logger.info(f"Loaded model from: {load_path}")
        return self.pipeline


class IncrementalTrainer:
    """Support incremental/online learning for new data."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None

    def load_existing_model(self) -> Pipeline:
        """Load the existing model for incremental training."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with open(self.model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        return self.pipeline

    def partial_fit(self, X, y):
        """Incrementally train on new data."""
        if self.pipeline is None:
            self.load_existing_model()

        # Get the classifier and do partial_fit
        classifier = self.pipeline.named_steps["classifier"]
        
        if hasattr(classifier, "partial_fit"):
            classifier.partial_fit(X, y)
            logger.info(f"Incremental training on {len(X)} samples")
        else:
            # For non-SGD classifiers, do full retrain on new data
            logger.warning("Classifier doesn't support partial_fit. Doing full retrain.")
            self.pipeline.fit(X, y)

    def update_with_new_data(self, new_data_path: str, labels_path: str):
        """Update model with new labeled data."""
        new_X = pd.read_csv(new_data_path)["text"].astype(str)
        new_y = pd.read_csv(labels_path)["label"].astype(str)
        
        self.partial_fit(new_X, new_y)
        self.save_model()

    def save_model(self):
        """Save the updated model."""
        with open(self.model_path, "wb") as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"Updated model saved to: {self.model_path}")


def train_large_dataset(
    data_path: str = None,
    output_model_path: str = None,
    use_sgd: bool = True,
    balance: bool = True
) -> Dict[str, Any]:
    """Main training function for large datasets."""
    data_path = data_path or DATA_PATH
    output_model_path = output_model_path or MODEL_PATH
    
    trainer = LargeDatasetTrainer()
    stats = trainer.train(data_path, use_sgd=use_sgd, balance=balance)
    trainer.save_model(output_model_path)
    
    return stats


def train_with_collected_data():
    """Train on collected social media data."""
    # Check if collected data exists
    if os.path.exists(LARGE_DATA_PATH):
        return train_large_dataset(LARGE_DATA_PATH)
    else:
        logger.warning(f"Collected data not found at: {LARGE_DATA_PATH}")
        return {"error": "No collected data found"}


if __name__ == "__main__":
    # Train on main dataset
    print("Training on cyberbullying dataset...")
    stats = train_large_dataset()
    print(f"Training complete: {stats}")