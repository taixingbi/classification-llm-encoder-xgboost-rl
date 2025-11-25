import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from data import texts, labels

nlp = spacy.load("en_core_web_sm")
def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        tokenizer=spacy_tokenizer,
        ngram_range=(1,2),
        min_df=1
    )),
    ("clf", LogisticRegression(max_iter=200))
])

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print(classification_report(y_test, preds))

sample = "Credit risk continues to worsen due to market volatility"
print("Prediction:", pipeline.predict([sample])[0])


