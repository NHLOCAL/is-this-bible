import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the dataset (assuming it is in UTF-8 encoding)
data = pd.read_csv('bible_talmud_data.csv', encoding='utf-8')

# Separate features (text) and labels (0, 1, or 2)
X = data['text']
y = data['label']

# Create a TF-IDF vectorizer with Hebrew tokenizer
vectorizer = TfidfVectorizer(tokenizer=word_tokenize, lowercase=True)

# Fit and transform the data with TF-IDF vectorizer
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=15)

# Create a Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear', C=2.0, probability=True)

# Train the SVM classifier on the training data
classifier.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Save the trained model and vectorizer to files
model_filename = "text_identification_model.pkl"
vectorizer_filename = "text_identification_vectorizer.pkl"
joblib.dump(classifier, model_filename)
joblib.dump(vectorizer, vectorizer_filename)
