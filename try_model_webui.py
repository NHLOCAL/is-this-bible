from flask import Flask, render_template, request
import webbrowser
import nltk
from nltk.corpus import stopwords
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer outside the routes for better performance
loaded_classifier = joblib.load("is_this_bible_model.pkl")
vectorizer = joblib.load("is_this_bible_vectorizer.pkl")

def parse_text(new_text):
    new_text_tfidf = vectorizer.transform([new_text])
    prediction = loaded_classifier.predict(new_text_tfidf)
    probabilities = loaded_classifier.predict_proba(new_text_tfidf)
    confidence_score = probabilities[0, 1]
    return 'תנ"ך' if prediction[0] == 1 else 'אחר', confidence_score

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence_score = None
    new_text = None

    if request.method == 'POST':
        new_text = request.form['new_text']
        if new_text:
            prediction, confidence_score = parse_text(new_text)
    return render_template('index.html', new_text=new_text, prediction=prediction, confidence_score=confidence_score)


if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)
