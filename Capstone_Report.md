# AI Python Capstone Project Report  
## Spam Classifier (Spam vs Not Spam)

---

## 1. Introduction
With the rapid growth of digital communication, spam messages have become a major problem.
Spam messages waste time, create security risks, and reduce productivity.

This project focuses on building a Machine Learning–based spam classifier using Python that can automatically identify whether a message is spam or not spam (ham).

---

## 2. Problem Statement
Manual identification of spam messages is inefficient and error-prone.
The objective of this project is to build an automated system that classifies text messages into:
- Spam
- Not Spam (Ham)

---

## 3. Objectives
- Understand text data preprocessing
- Apply machine learning techniques for classification
- Build a spam detection model using Python
- Evaluate model performance
- Create a real-world AI solution

---

## 4. Dataset Description
The project uses the **SMS Spam Collection Dataset**, which contains labeled messages.

- Labels:
  - spam
  - ham
- Total messages: 5,000+
- Data format: CSV

---

## 5. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- VS Code
- Git & GitHub

---

## 6. Methodology

### 6.1 Data Preprocessing
- Removed unnecessary columns
- Renamed columns for clarity
- Converted text labels into numerical values
- Cleaned and prepared text data

### 6.2 Feature Extraction
- Used **TF-IDF Vectorization**
- Converted text into numerical features suitable for ML models

### 6.3 Model Selection
- Used **Multinomial Naive Bayes**
- Suitable for text classification problems

### 6.4 Model Training
- Dataset split into training and testing sets (80/20)
- Model trained on training data

---

## 7. Model Evaluation
- Evaluation Metric: Accuracy
- Achieved accuracy of approximately **97%**
- Model performs well in identifying spam messages

---

## 8. Results
The trained model successfully classifies messages as spam or not spam.
The system works efficiently for real-time text input.

Example:
- "FREE entry into win cash" → Spam
- "Are we meeting tomorrow?" → Not Spam

---

## 9. Project Structure

spam_classifier/
├── data/
├── src/
│ ├── train.py
│ ├── predict.py
│ └── helper_functions.py
├── results/
├── README.md
├── requirements.txt
└── Capstone_Report.md


---

## 10. Conclusion
This project demonstrates the practical application of AI and machine learning in solving a real-world problem.
The spam classifier provides accurate predictions and can be extended further using advanced models.

---

## 11. Future Enhancements
- Use Deep Learning models (LSTM, BERT)
- Web interface using Streamlit or Flask
- Email spam detection
- Multilingual support

---

## 12. Author
Name: Nandesh  
Project Type: AI Python Capstone Project

---

## 13. Declaration
I hereby declare that this project is my original work and has not been copied from any source.
