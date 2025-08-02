from flask import Flask, request, render_template
import joblib
import re
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_detection_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    stemmer = SnowballStemmer("english")
    return text

# Predict function
def predict_spam_or_ham(text):
    clean_text = preprocess_text(text)
    text_vector = vectorizer.transform([clean_text])
    prediction = model.predict(text_vector)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sms = request.form['sms']
    result = predict_spam_or_ham(sms)
    return render_template('index.html', prediction=result, sms_text=sms)

if __name__ == '__main__':
    app.run(debug=True)
