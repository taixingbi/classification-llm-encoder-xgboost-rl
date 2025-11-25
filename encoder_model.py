from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from torch.optim import AdamW   # <-- FIXED

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import numpy as np

MODEL_NAME = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------
# 1. Example dataset  (replace with Moody’s real data)
# ----------------------------------------
from data import texts, labels

# Encode labels to integers
label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}

y = [label2id[x] for x in labels]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.33, random_state=42, stratify=y
)

# ----------------------------------------
# 2. Tokenizer
# ----------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoded = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx])
        }

train_ds = TextDataset(X_train, y_train)
test_ds  = TextDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=4)

# ----------------------------------------
# 3. Encoder Model
# ----------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# ----------------------------------------
# 4. Train Loop
# ----------------------------------------
model.train()
for epoch in range(2):    # small number for demo
    for batch in train_loader:
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} finished.")

# ----------------------------------------
# 5. Evaluation
# ----------------------------------------
model.eval()
preds = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        ).logits
        
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(predictions)
        true_labels.extend(batch["labels"].cpu().numpy())

# Convert predictions back to text labels
pred_labels = [id2label[p] for p in preds]
true_labels_text = [id2label[y] for y in true_labels]

print("=== Option C: Encoder Model (DistilBERT) ===")
print(classification_report(true_labels_text, pred_labels))
