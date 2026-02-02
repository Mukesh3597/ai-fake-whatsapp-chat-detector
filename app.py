import os
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# -----------------------------
# Load Chat Model (ONCE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chat_detector_model.pkl")

chat_model = joblib.load(MODEL_PATH)


def predict_with_details(text: str):
    # 1) Prediction
    pred = chat_model.predict([text])[0]

    # 2) Probabilities
    proba = chat_model.predict_proba([text])[0]
    classes = list(chat_model.classes_)
    prob_map = {classes[i]: round(float(proba[i]) * 100, 2) for i in range(len(classes))}

    # 3) Confidence
    confidence = prob_map.get(pred, round(float(proba.max()) * 100, 2))

    # 4) Explanation (simple)
    if pred == "fake":
        explain = "Scam/Threat pattern detected (OTP/KYC/block/click/refund type words)."
    elif pred == "suspicious":
        explain = "Forward/Chain style detected (share/forward/urgent/breaking type words)."
    else:
        explain = "Looks like normal human conversation."

    # 5) Top keywords (TF-IDF)
    keywords = []
    try:
        tfidf = chat_model.named_steps["tfidf"]
        X_vec = tfidf.transform([text])
        feature_names = tfidf.get_feature_names_out()
        row = X_vec.toarray()[0]

        top_idx = row.argsort()[::-1]
        for i in top_idx:
            if row[i] <= 0:
                break
            w = feature_names[i]
            if len(w) < 3:
                continue
            keywords.append(w)
            if len(keywords) == 5:
                break
    except Exception:
        keywords = []

    return pred, confidence, keywords, prob_map, explain


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        result=None,
        confidence=None,
        keywords=[],
        prob_map={},
        explain=None,
        chat=""
    )


@app.route("/check-chat", methods=["POST"])
def check_chat():
    chat_text = request.form.get("chat", "").strip()

    result = confidence = explain = None
    keywords = []
    prob_map = {}

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


# Local run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
