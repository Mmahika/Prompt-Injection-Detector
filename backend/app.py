from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)
CORS(app)

vectorizer = joblib.load('vectorizer.pkl')
baseline_model = joblib.load('model.pkl')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
distilbert_model.eval()

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

    inputs = tokenizer(text, return_tensors='pt',
                      max_length=128, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        injection_prob = probs[0][1].item()
        safe_prob = probs[0][0].item()

    if injection_prob > 0.85:
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
