from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# ✅ Always load model using absolute path (works on Render + local)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chat_detector_model.pkl")

model = joblib.load(MODEL_PATH)


def predict_with_details(text: str):
    # 1) Prediction
    pred = model.predict([text])[0]

    # 2) Probabilities
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    prob_map = {classes[i]: round(float(proba[i]) * 100, 2) for i in range(len(classes))}

    # 3) Confidence = predicted class probability
    confidence = prob_map.get(pred, round(float(proba.max()) * 100, 2))

    # 4) Explanation (simple rules)
    if pred == "fake":
        explain = "Scam/Threat pattern detected (OTP/KYC/block/click/refund type words)."
    elif pred == "suspicious":
        explain = "Forward/Chain style detected (share/forward/urgent/breaking type words)."
    else:
        explain = "Looks like normal human conversation."

    # 5) Top keywords from TF-IDF (from input text)
    try:
        tfidf = model.named_steps["tfidf"]
        X_vec = tfidf.transform([text])
        feature_names = tfidf.get_feature_names_out()
        row = X_vec.toarray()[0]

        top_idx = row.argsort()[::-1]
        keywords = []
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

    # ✅ Return 5 things
    return pred, confidence, keywords, prob_map, explain


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    keywords = []
    prob_map = {}
    explain = None
    chat_text = ""

    if request.method == "POST":
        chat_text = request.form.get("chat", "").strip()
        if chat_text:
            # ✅ unpack 5 values
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
    app.run(debug=True)
