# Prompt Injection Detector

A machine learning system that detects prompt injection attacks against LLMs.

## Live Demo
- Frontend: https://mmahika.github.io/Prompt-Injection-Detector/frontend/index.html
- API: https://prompt-injection-detector-xaos.onrender.com

## What it does
Classifies user prompts as **safe** or **injection attack** in real time.

## Models
| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| TF-IDF + Logistic Regression | 95% | 0.96 | 0.94 |
| DistilBERT (fine-tuned) | 98% | 0.98 | 0.98 |

## Project Structure
- `dataset/` — labeled prompt injection dataset (4,391 examples)
- `notebooks/` — data exploration, baseline model, DistilBERT training
- `backend/` — Flask API serving the model
- `frontend/` — web interface for testing prompts

## Tech Stack
Python, scikit-learn, HuggingFace Transformers, PyTorch, Flask

## Run locally
```bash
pip install -r requirements.txt
python backend/app.py
```
