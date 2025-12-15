import tempfile
import os
import pandas as pd
from src import helper_functions as hf


def run():
    # preprocess test
    texts = ["Hello, WORLD!", "<p>Spam</p>", "Visit: http://example.com!!!"]
    cleaned = hf.preprocess_texts(texts)
    assert cleaned[0] == "hello world"
    assert cleaned[1] == "spam"
    assert "visit" in cleaned[2]

    # vectorize test
    texts2 = ["this is spam", "not spam here"]
    X, vec = hf.vectorize_text(texts2)
    assert X.shape[0] == 2
    assert hasattr(vec, "vocabulary_")

    new_texts = ["spam here"]
    X2, _ = hf.vectorize_text(new_texts, vectorizer=vec)
    assert X2.shape[0] == 1

    # load_data test
    import csv
    tmpdir = tempfile.mkdtemp()
    p = os.path.join(tmpdir, 't.csv')
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    df.to_csv(p, index=False)
    texts_out, labels = hf.load_data(p)
    assert texts_out.tolist() == ["a", "b"]
    assert labels.tolist() == [0, 1]

    # missing text column should raise
    p2 = os.path.join(tmpdir, 't2.csv')
    df2 = pd.DataFrame({"body": ["x"]})
    df2.to_csv(p2, index=False)
    try:
        hf.load_data(p2)
        raise SystemExit('Expected ValueError for missing text column')
    except ValueError:
        pass

    print('All helper function checks passed')


if __name__ == '__main__':
    run()
