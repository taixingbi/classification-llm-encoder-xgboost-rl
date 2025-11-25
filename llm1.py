from openai import OpenAI
from data import texts, labels
from sklearn.metrics import classification_report
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are a financial text classifier.
You read analyst-style sentences about companies and classify the RISK SENTIMENT
into one of two labels:

- positive  = outlook improving, upgrade, lower risk
- negative  = outlook deteriorating, downgrade, higher risk

Return EXACTLY one word: positive or negative.
No explanations.
"""

def classify_text(text: str) -> str:
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Text: "{text}"\nLabel:'},
        ],
    )
    label = resp.output[0].content[0].text.strip().lower()
    return label


if __name__ == "__main__":
    preds = []

    for t in texts:
        pred = classify_text(t)
        preds.append(pred)
        print(f"{t} -> {pred}")

    print("\n=== Classification Report (Prompt-Based LLM) ===")
    print(classification_report(labels, preds))
