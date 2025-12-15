"""Training CLI for spam classifier."""

from pathlib import Path
from typing import Optional

import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from src.helper_functions import load_data, preprocess_texts, vectorize_text


def train(data_path: Optional[str] = 'data/spam.csv', model_out: str = 'models/spam_model.pkl', vectorizer_out: str = 'models/vectorizer.pkl') -> None:
    """Train a simple TF-IDF + MultinomialNB model and save artifacts.

    Args:
        data_path: Path to CSV dataset.
        model_out: Path to save trained model.
        vectorizer_out: Path to save fitted vectorizer.
    """
    # Try common column names for text and label
    text_cols = ['text', 'message', 'v2', 'body']
    label_cols = ['label', 'v1', 'class']

    # Load dataset; try different column names
    df_paths = [data_path]
    texts = None
    labels = None
    for tp in df_paths:
        for tcol in text_cols:
            for lcol in label_cols:
                try:
                    texts, labels = load_data(tp, text_column=tcol, label_column=lcol)
                    # require both text and label columns present
                    if labels is None:
                        continue
                    break
                except (FileNotFoundError, ValueError):
                    continue
            if texts is not None and labels is not None:
                break
        if texts is not None and labels is not None:
            break

    if texts is None:
        raise ValueError("Could not find suitable text/label columns in dataset")

    # Some datasets have labels like 'ham'/'spam' in text; map them
    if labels is not None and labels.dtype == object:
        labels = labels.map({'ham': 0, 'spam': 1}).astype(int)

    # Preprocess
    cleaned = preprocess_texts(texts)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(cleaned, labels, test_size=0.2, random_state=42)

    # Vectorize
    X_train_vec, vectorizer = vectorize_text(X_train)
    X_test_vec, _ = vectorize_text(X_test, vectorizer=vectorizer)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Save artifacts
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    Path(vectorizer_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_out)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


def main():
    train()


if __name__ == "__main__":
    main()
