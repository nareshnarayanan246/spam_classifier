# Spam Classifier

[![Python Tests](https://github.com/OWNER/REPO/actions/workflows/python-tests.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/python-tests.yml) [![Tests Status](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/OWNER/REPO/actions/workflows/python-tests.yml)

Small spam classifier example using TF-IDF + MultinomialNB.

> Tip: Replace `OWNER/REPO` in the GitHub Actions badge URL with your GitHub repository owner/name to enable a dynamic status badge.

## Setup

Create a virtual environment and install dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

Train a model using the default data file (`data/spam.csv`):

```powershell
python -m src.train
```

Or provide a custom dataset path and artifact outputs in Python:

```python
from src import train
train.train('data/spam.csv', model_out='models/spam_model.pkl', vectorizer_out='models/vectorizer.pkl')
```

## Predict

Use the CLI to predict one or more texts:

```powershell
python -m src.predict -t "You won a prize" "hello friend"
```

Or call `predict.predict([...])` from Python code.

## Tests

Run tests with `pytest`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Notes

- Default data file is `data/spam.csv`. The loader will attempt common column names (`text`, `message`, `v2`, etc.).
- Artifacts are saved to the `models/` directory by default.
