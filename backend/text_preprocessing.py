import re

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def preprocess_text(text: str) -> str:
    """Lowercase, remove punctuation, remove stopwords, and tokenize."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
    tokens = tokenizer.tokenize(text)
    stops = set(stopwords.words("english"))
    filtered = [token for token in tokens if token not in stops]
    return " ".join(filtered)


def tokenize_text(text: str) -> list[str]:
    """Split cleaned text into tokens for TF-IDF."""
    return text.split()
