import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

df = pd.read_csv("data/processed/apple_event_stock_labeled.csv")

X = df["text"].astype(str)
y = df["stock_movement"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/stock_movement_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("✅ Model and vectorizer saved in 'models/' folder")
