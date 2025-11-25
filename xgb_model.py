from data import texts, labels

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgb_model import XGBClassifier

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# TF-IDF + XGBoost
xgb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),
        min_df=1
    )),
    ("xgb", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )),
])

xgb_pipeline.fit(X_train, y_train)
pred = xgb_pipeline.predict(X_test)

print("=== TF-IDF + XGBoost ===")
print(classification_report(
    le.inverse_transform(y_test),
    le.inverse_transform(pred)
))
