from sys import argv
#import re
#import nltk
#from nltk.corpus import stopwords
import joblib


"""
# Remove punctuation and special characters
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Function to remove custom stop words from text
def remove_custom_stopwords(text):
    hebrew_stopwords = set(stopwords.words('hebrew'))
    additional_stopwords = {'אני', 'אתה', 'את', 'אנחנו', 'אתם', 'אתן', 'הם', 'הן'}
    hebrew_stopwords.update(additional_stopwords)
    return ' '.join(word for word in text.split() if word not in hebrew_stopwords)
    
    
# Preprocess the new text (remove punctuation and custom stop words)
# אם רוצים להחזיר את הפונקצייה הלא פעילה יש להעביר את המשתנה אחרי המשתנה new_text
new_text_cleaned = remove_custom_stopwords(remove_punctuation(new_text))
"""


# Load the trained model from the file
loaded_classifier = joblib.load("is_this_bible_model.pkl")

# Load the TF-IDF vectorizer used for training
vectorizer = joblib.load("is_this_bible_vectorizer.pkl")

def parse_text(new_text):
    # Transform the new text using the TF-IDF vectorizer
    new_text_tfidf = vectorizer.transform([new_text])

    # Make predictions on the new text
    prediction = loaded_classifier.predict(new_text_tfidf)

    # Get the confidence score for the predicted class
    probabilities = loaded_classifier.predict_proba(new_text_tfidf)
    confidence_score = probabilities[0, 1]  # The confidence score for class "Bible" (index 1)

    # Print the prediction and the confidence score
    print(f"Text: {new_text} | Prediction: {'Bible' if prediction[0] == 1 else 'Other'} | Confidence Score: {confidence_score:.4f}")


text_list = [
'אני יושב פה בשקט ומקלל את העובדה שחלק מהתוכנות שאני מתחזק קשורה לפייתון 2.4, שאין לה את זה',
'כמה יפה ונאה כששומעים השירה שלהם',
'והיה בעת ההיא אחפש את ירושלים בנרות והודעתיה את כל תועבותיה',
'והיא שעמדה לאבותינו ולנו שלא אחד בלבד עמד עלינו לכלותינו',
'אני הסתכלתי לשמים אתה צללת במים',
'הצב הוא בעל חיים שחי בים וביבשה',
'והיה הנשאר בציון והנותר בירושלים קדוש יאמר לו',
'שיר השירים אשר לשלמה',
'ישקני מנשיקות פיהו כי טובים דודיך מיין',
'והיה רק מלא שמחה וחדוה תמיד כשהיה גומר המנעל ומן הסתם היה לו שלשה קצוות',
'זה מעשה שלו וזה מעשה שלי ועוד מה לנו לדבר מאחרים',
'דודי ירד לגנו לערוגות הבושם לרעות בגנים וללקוט שושנים',
'וימרו בי בית ישראל במדבר בחקותי לא הלכו ואת משפטי מאסו אשר יעשה אתם האדם וחי בהם',
'זה לא משנה אופניים נעליים העיקר זה בחיים',
'זכור את יום השבת לקדשו',
'וישלח יעקב מלאכים לפניו אל עשיו אחיו',
'לך לך מארצך וממולדתך ומבית אביך',
'עדכון :דור לדור תנ"ך ,מאורעות בזמן התנ"ך קרדיט']

if argv[1:]:
    new_text = argv[1]
    parse_text(new_text)
    
else:
    for new_text in text_list:
        parse_text(new_text)
