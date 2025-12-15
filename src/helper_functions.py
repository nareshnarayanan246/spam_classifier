import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(path: Union[str, Path], text_column: str = 'text', label_column: Optional[str] = 'label') -> Tuple[pd.Series, Optional[pd.Series]]:
    """Load CSV and return (texts, labels).

    Args:
        path: Path to a CSV file (string or Path).
        text_column: Name of the column that contains text data.
        label_column: Name of the label column (optional).

    Returns:
        A tuple of (`texts`, `labels`) where `labels` may be ``None`` if not present.

    Raises:
        FileNotFoundError: if the resolved file does not exist.
        ValueError: if `text_column` is not present in the CSV.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Data file not found at: {p}")

    df = pd.read_csv(p)
    if text_column not in df.columns:
        raise ValueError(f"Expected text column '{text_column}' in CSV")
    texts = df[text_column].astype(str)
    labels = df[label_column] if label_column and label_column in df.columns else None
    return texts, labels


def preprocess_texts(texts: Iterable[str]) -> list:
    """Basic text preprocessing: lowercase, remove HTML/tags, non-alphanumeric chars, collapse whitespace."""
    cleaned = []
    for t in texts:
        s = str(t).lower()
        s = re.sub(r'<[^>]+>', ' ', s)
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        cleaned.append(s)
    return cleaned


def vectorize_text(texts: Iterable[str], vectorizer: Optional[TfidfVectorizer] = None) -> Tuple[csr_matrix, TfidfVectorizer]:
    """Fit-and-transform texts with TF-IDF or transform using an existing vectorizer.

    If `vectorizer` is None, a new `TfidfVectorizer` will be fitted and returned.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)
    return X, vectorizer
