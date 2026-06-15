# Prompt Injection Detector

A machine learning system that detects prompt injection attacks against LLMs.

## What it does
Classifies user prompts as **safe** or **injection attack** in real time.

## Models
| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| TF-IDF + Logistic Regression | 95% | 0.96 | 0.94 |
| DistilBERT (fine-tuned) | TBD | TBD | TBD |

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
