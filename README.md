# NLP Classification Project

```
python3 -m venv my_env
source my_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

This project demonstrates multiple approaches to text classification using a finance-focused,
Moody’s-style dataset. The goal is to compare four major NLP modeling families:

1. **LLM Prompt-Based Classification** (zero-shot / few-shot / RAG-style)
2. **Transformer Encoder Fine-Tuning** (BERT / DistilBERT)
3. **XGBoost Classification** (with TF-IDF or embeddings)
4. **Logistic Regression Baseline** (TF-IDF)

The project includes a small synthetic credit-risk dataset that imitates analyst-style
sentences used for rating decisions, credit opinions, and outlook analysis.

## 🚀 Features

### **1. Prompt-Based LLM Classification (with RAG-Few-Shot)**
- Uses OpenAI `responses` API (`gpt-4.1-mini`)
- Retrieves similar labeled samples using TF-IDF cosine similarity
- Injects retrieved examples as few-shot demonstrations
- Zero training required

### **2. Transformer Encoder Fine-Tuning**
- Uses `AutoModelForSequenceClassification` (DistilBERT)
- PyTorch-based training loop
- GPU-friendly but can run CPU with smaller batch size
- Best supervised accuracy

### **3. XGBoost Classifier**
- Strong classical ML baseline
- Works well with TF-IDF
- Fast training and low latency
- Great for medium datasets

### **4. Logistic Regression Baseline**
- TF-IDF + LogisticRegression
- Very fast, interpretable
- Good first model to validate the dataset

## 📁 Project Structure

```
nlp/
│
├── data.py                     # Moody’s-style synthetic dataset
│
├── llm_rag_few_shot.py    # LLM + RAG few-shot classifier
├── encoder_model.py            # DistilBERT fine-tuning
├── xgb_model.py                # XGBoost classifier
├── lr_model.py                 # Logistic Regression baseline
│
└── utils/                      # (optional utilities)
```

## 📊 Model Comparison

| Category | **LLM (Prompt / RAG)** | **Encoder (BERT / DistilBERT)** | **XGBoost (TF-IDF / Embeddings)** | **Logistic Regression (TF-IDF)** |
|---------|------------------------|----------------------------------|-----------------------------------|----------------------------------|
| **Training Required** | ❌ None | ✅ Yes (fine-tuning) | ✅ Yes | ✅ Yes |
| **Data Needed** | ⭐ Very little (few-shot) | ⭐⭐ Medium (1k–100k) | ⭐ Small–Medium | ⭐ Small–Medium |
| **Understands Context** | ⭐⭐⭐⭐ Excellent (reasoning, credit nuance) | ⭐⭐⭐⭐ Strong contextual | ⭐⭐ Medium (bag-of-words) | ⭐ Weak (linear) |
| **Interpretability** | ⭐ Low | ⭐⭐ Medium | ⭐⭐ Medium | ⭐⭐⭐⭐ High |
| **Latency** | ❌ Slow (API call) | Medium (GPU/CPU) | Fast | ⚡ Very fast |
| **Compute Cost** | $$$ Highest | $$ Moderate | $ Low | $ Lowest |
| **Deployment Complexity** | Hard (API / LLM infra) | Medium | Easy | ⭐ Very easy |
| **Few-Shot Performance** | ⭐⭐⭐⭐ Excellent | ⭐⭐ Needs training | ⭐ Medium | ⭐ Weak |
| **Large Dataset Performance** | Good but $$$ | ⭐⭐⭐⭐ Best | ⭐⭐⭐⭐ Strong | ⭐⭐ Limited |
| **Captures Financial Nuance** | ⭐⭐⭐⭐ Best | ⭐⭐⭐ Strong | ⭐⭐ Medium | ⭐ Low |
| **Determinism** | Low–Medium | High | High | Very High |
| **Best Use Cases** | Reasoning text, credit outlook, narrative risk | Rated text, credit sentiment, NER | Production classification, low-latency | Baseline model, sanity checks |

## 🔧 Setup

### 1. Install dependencies

```
mamba install -y scikit-learn
mamba install -y xgboost
mamba install -y transformers
mamba install -y torch
mamba install -y openai
mamba install -y spacy
python -m spacy download en_core_web_sm
```
