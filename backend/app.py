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
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = round(max(proba) * 100, 2)
    result = 'injection' if pred == 1 else 'safe'

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
