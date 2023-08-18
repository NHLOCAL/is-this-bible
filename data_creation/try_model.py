from sys import argv
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the trained model from the file
loaded_classifier = joblib.load("text_identification_model.pkl")

# Load the TF-IDF vectorizer used for training
vectorizer = joblib.load("text_identification_vectorizer.pkl")

# Define labels for your categories
categories = {0: 'Other', 1: 'Bible', 2: 'Talmud'}

def parse_text(new_text):
    # Transform the new text using the TF-IDF vectorizer
    new_text_tfidf = vectorizer.transform([new_text])

    # Make predictions on the new text
    prediction = loaded_classifier.predict(new_text_tfidf)

    # Get the confidence score for the predicted class
    probabilities = loaded_classifier.predict_proba(new_text_tfidf)
    confidence_score = probabilities[0, 1]  # Confidence score for class "Bible" (index 1)

    # Determine the predicted category label
    predicted_category = categories[prediction[0]]

    # Print the prediction and the confidence score
    print(f"Text: {new_text} | Prediction: {predicted_category} | Confidence Score: {confidence_score:.4f}")


text_list = [
'כמה יפה ונאה כששומעים השירה שלהם',
'חדשות הערב: שלושה אנשים נצםו טובעים בכינרת',
'והיה בעת ההיא אחפש את ירושלים בנרות והודעתיה את כל תועבותיה',
'ויאמר משה אל בני ישראל',
'דאמר נשיא מביא שעיר תו הא דתנן',
'אמר ליה אביי לרב זעירא',
'ואיהו לא קא יהיב שעורא במשכא',]


if argv[1:]:
    new_text = argv[1]
    parse_text(new_text)
else:
    for new_text in text_list:
        parse_text(new_text)
