import argparse
from typing import List

import joblib

from src.helper_functions import preprocess_texts


def predict(texts: List[str], model_path: str = 'models/spam_model.pkl', vectorizer_path: str = 'models/vectorizer.pkl'):
	model = joblib.load(model_path)
	vectorizer = joblib.load(vectorizer_path)

	cleaned = preprocess_texts(texts)
	text_vector = vectorizer.transform(cleaned)
	prediction = model.predict(text_vector)
	return prediction


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--text', '-t', nargs='+', default=["Congratulations! You won a free prize"])
	args = parser.parse_args()
	preds = predict(args.text)
	for p in preds:
		print("Prediction:", p)


if __name__ == '__main__':
	main()
