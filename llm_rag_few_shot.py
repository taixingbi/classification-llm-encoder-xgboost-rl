from openai import OpenAI
from data import texts, labels
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# 1. System prompt
# -------------------------------
SYSTEM_PROMPT = """
You are a financial text classifier.
You read analyst-style sentences about companies and classify the RISK SENTIMENT
into one of two labels:

- positive  = outlook improving, upgrade, lower risk
- negative  = outlook deteriorating, downgrade, higher risk

You will be given several EXAMPLES with their correct labels.
Then you will be given a NEW text to classify.

Return EXACTLY one word: positive or negative.
No explanations.
"""

# -------------------------------
# 2. Build retrieval index (TF-IDF)
#    This is our "RAG" store of labeled examples
# -------------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
)

X = vectorizer.fit_transform(texts)  # shape: [n_examples, vocab]


def retrieve_examples(query: str, k: int = 3):
    """
    Retrieve top-k most similar labeled examples (RAG-style)
    using TF-IDF cosine similarity.
    """
    q_vec = vectorizer.transform([query])        # [1, vocab]
    sims = cosine_similarity(q_vec, X)[0]        # [n_examples]
    top_idx = np.argsort(sims)[::-1][:k]         # highest similarity first

    examples = []
    for idx in top_idx:
        examples.append((texts[idx], labels[idx]))
    return examples


def build_few_shot_block(query: str, k: int = 3) -> str:
    """
    Build a few-shot examples text block from retrieved examples.
    """
    examples = retrieve_examples(query, k=k)
    lines = []
    for t, lab in examples:
        lines.append(f'Text: "{t}"\nLabel: {lab}\n')
    return "\n".join(lines)


# -------------------------------
# 3. Prompt-based + RAG few-shot classifier
# -------------------------------
def classify_text(text: str, k: int = 3) -> str:
    few_shot_block = build_few_shot_block(text, k=k)

    user_content = f"""
Here are some labeled examples:

{few_shot_block}

Now classify this new text:

Text: "{text}"
Label:
""".strip()

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    label = resp.output[0].content[0].text.strip().lower()
    # normalize to exactly 'positive' or 'negative' just in case
    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    return label  # fallback, in case model misbehaves


# -------------------------------
# 4. Demo & evaluation
# -------------------------------
if __name__ == "__main__":
    preds = []

    for t in texts:
        pred = classify_text(t, k=3)  # RAG-style few-shot
        preds.append(pred)
        print(f"{t} -> {pred}")

    print("\n=== Classification Report (RAG Few-Shot LLM) ===")
    print(classification_report(labels, preds))
