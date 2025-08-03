from flask import Flask, request, jsonify, render_template
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import SnowballStemmer
import numpy as np
import joblib

model=joblib.load('model.pkl')
vectorizer=joblib.load('vectorizer.pkl')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    stemmer = SnowballStemmer("english")
    
    # Apply stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get text from form data
    text = request.form.get('text', '')
    
    if not text:
        return render_template('index.html', prediction_text='Please enter some text to classify')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])

    # Make prediction
    prediction = model.predict(text_tfidf)
    
    # Get prediction result
    output = 'Spam' if prediction[0] == 1 else 'Ham'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)