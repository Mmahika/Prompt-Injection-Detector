from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

base_dir = os.path.dirname(os.path.abspath(__file__))
vectorizer = joblib.load(os.path.join(base_dir, 'vectorizer.pkl'))
model = joblib.load(os.path.join(base_dir, 'model.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    if len(text.split()) < 4:
        return jsonify({
            'text': text,
            'prediction': 'safe',
            'confidence': 99.0
        })

    features = vectorizer.transform([text])
    proba = model.predict_proba(features)[0]
    injection_prob = proba[1]
    safe_prob = proba[0]

    if injection_prob > 0.80:
        result = 'injection'
        confidence = round(injection_prob * 100, 2)
    else:
        result = 'safe'
        confidence = round(safe_prob * 100, 2)

    return jsonify({
        'text': text,
        'prediction': result,
        'confidence': confidence
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
