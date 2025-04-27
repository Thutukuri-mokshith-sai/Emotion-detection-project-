# Import Libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load Model and Vectorizer
with open('/emotion_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Download stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing Function
def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# API Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request
        data = request.get_json()
        text = data['text']
        
        # Preprocess
        text_clean = preprocess(text)
        text_vec = vectorizer.transform([text_clean])

        # Predict
        prediction = model.predict(text_vec)
        emotion = prediction[0]

        # Send response
        return jsonify({'emotion': emotion})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
