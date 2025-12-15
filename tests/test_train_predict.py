import os
from pathlib import Path
import pandas as pd
from src import train as training
from src import predict as predicting


def test_train_and_predict(tmp_path):
    # Create small dataset
    df = pd.DataFrame({
        'v1': ['ham', 'spam', 'ham'],
        'v2': ['hello friend', 'win money now', 'see you later']
    })
    p = tmp_path / 'small.csv'
    df.to_csv(p, index=False)

    model_path = tmp_path / 'models' / 'spam_model.pkl'
    vec_path = tmp_path / 'models' / 'vectorizer.pkl'

    # Train
    training.train(str(p), str(model_path), str(vec_path))

    assert model_path.exists()
    assert vec_path.exists()

    # Predict
    preds = predicting.predict(['win money now'], model_path=str(model_path), vectorizer_path=str(vec_path))
    assert len(preds) == 1
    # prediction should be 0 or 1
    assert preds[0] in (0, 1)
