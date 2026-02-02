# 1) जरूरी libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# 2) Dataset load
data = pd.read_csv("data.csv")

# 3) Features (X) और Labels (y)
X = data["text"]
y = data["label"]

# 4) Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) ML Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

# 6) Model train
model.fit(X_train, y_train)

# 7) Accuracy check
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# 8) Model save
joblib.dump(model, "chat_detector_model.pkl")

print("Model trained and saved successfully!")
