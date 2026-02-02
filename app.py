import os
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# -------------------
# Paths
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "chat_detector_model.pkl")

# -------------------
# Load model ONCE
# -------------------
chat_model = joblib.load(MODEL_PATH)

# -------------------
# Prediction logic
# -------------------
def predict_with_details(text: str):
    pred = chat_model.predict([text])[0]

    proba = chat_model.predict_proba([text])[0]
    classes = list(chat_model.classes_)
    prob_map = {
        classes[i]: round(float(proba[i]) * 100, 2)
        for i in range(len(classes))
    }

    confidence = prob_map.get(pred, round(float(proba.max()) * 100, 2))

    if pred == "fake":
        explain = "Scam/Threat pattern detected (OTP/KYC/block/click type words)."
    elif pred == "suspicious":
        explain = "Forward/Chain style message detected."
    else:
        explain = "Looks like a normal human conversation."

    # TF-IDF keywords
    tfidf = chat_model.named_steps["tfidf"]
    X_vec = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()
    row = X_vec.toarray()[0]

    top_idx = row.argsort()[::-1]
    keywords = []
    for i in top_idx:
        if row[i] <= 0:
            break
        w = feature_names[i]
        if len(w) >= 3:
            keywords.append(w)
        if len(keywords) == 5:
            break

    return pred, confidence, keywords, prob_map, explain


# -------------------
# Routes
# -------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = confidence = explain = None
    keywords = []
    prob_map = {}
    chat_text = ""

    if request.method == "POST":
        chat_text = request.form.get("chat", "").strip()
        if chat_text:
            result, confidence, keywords, prob_map, explain = predict_with_details(chat_text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        keywords=keywords,
        prob_map=prob_map,
        explain=explain,
        chat=chat_text
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
