import io
import pandas as pd
import pytest
from src import helper_functions as hf


def test_preprocess_texts():
    texts = ["Hello, WORLD!", "<p>Spam</p>", "Visit: http://example.com!!!"]
    cleaned = hf.preprocess_texts(texts)
    assert cleaned[0] == "hello world"
    assert cleaned[1] == "spam"
    assert "visit" in cleaned[2]


def test_vectorize_text_fit_and_transform():
    texts = ["this is spam", "not spam here"]
    X, vec = hf.vectorize_text(texts)
    assert X.shape[0] == 2
    assert hasattr(vec, "vocabulary_")

    # Transform new texts
    new_texts = ["spam here"]
    X2, _ = hf.vectorize_text(new_texts, vectorizer=vec)
    assert X2.shape[0] == 1


def test_load_data(tmp_path):
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)

    texts, labels = hf.load_data(str(p))
    assert texts.tolist() == ["a", "b"]
    assert labels.tolist() == [0, 1]

    # Missing text column
    df2 = pd.DataFrame({"body": ["x"]})
    p2 = tmp_path / "t2.csv"
    df2.to_csv(p2, index=False)
    with pytest.raises(ValueError):
        hf.load_data(str(p2))
