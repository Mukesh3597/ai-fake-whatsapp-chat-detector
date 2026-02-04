import os
import joblib
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# -----------------------------
# Optional TensorFlow (server pe install ho to image model चलेगा)
# -----------------------------
try:
    import tensorflow as tf
    TF_OK = True
except Exception:
    tf = None
    TF_OK = False

app = Flask(__name__)

# -----------------------------
# Paths / Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

CHAT_MODEL_PATH = os.path.join(MODELS_DIR, "chat_detector_model.pkl")
IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "ai_image_detector.h5")

IMG_SIZE = (224, 224)

# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------
# Load Models (Safe)
# -----------------------------
chat_model = None
chat_error = None
if os.path.exists(CHAT_MODEL_PATH):
    try:
        chat_model = joblib.load(CHAT_MODEL_PATH)
    except Exception as e:
        chat_error = f"Chat model load error: {e}"
else:
    chat_error = "Chat model file not found in /models."

image_model = None
image_error = None
if TF_OK and os.path.exists(IMAGE_MODEL_PATH):
    try:
        image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    except Exception as e:
        image_error = f"Image model load error: {e}"
else:
    if not TF_OK:
        image_error = "TensorFlow not installed on server."
    elif not os.path.exists(IMAGE_MODEL_PATH):
        image_error = "Image model file not found in /models."


# -----------------------------
# Chat Prediction
# -----------------------------
def predict_with_details(text: str):
    if chat_model is None:
        return None, None, [], {}, chat_error or "Chat model not available."

    pred = chat_model.predict([text])[0]

    confidence = None
    prob_map = {}

    # If model supports probabilities
    if hasattr(chat_model, "predict_proba"):
        proba = chat_model.predict_proba([text])[0]
        classes = list(chat_model.classes_)
        prob_map = {classes[i]: round(float(proba[i]) * 100, 2) for i in range(len(classes))}
        confidence = prob_map.get(pred, round(float(np.max(proba)) * 100, 2))
    else:
        confidence = 0

    if pred == "fake":
        explain = "Scam/Threat pattern detected (OTP/KYC/block/click/refund type words)."
    elif pred == "suspicious":
        explain = "Forward/Chain style detected (share/forward/urgent/breaking type words)."
    else:
        explain = "Looks like normal human conversation."

    # Top keywords (TF-IDF) if pipeline has tfidf step
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
        pass

    return pred, confidence, keywords, prob_map, explain


# -----------------------------
# Image Prediction
# -----------------------------
def predict_image(file_path: str):
    if not TF_OK or image_model is None:
        return "Image detector disabled on server (TensorFlow not installed).", None, None

    img = Image.open(file_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    y = image_model.predict(x, verbose=0)

    # sigmoid (1 output)
    if y.shape[-1] == 1:
        prob_fake = float(y[0][0])
        label = "FAKE (AI-Generated)" if prob_fake >= 0.5 else "REAL"
        conf = round((prob_fake if prob_fake >= 0.5 else (1 - prob_fake)) * 100, 2)
        scores = {
            "real": round((1 - prob_fake) * 100, 2),
            "fake": round(prob_fake * 100, 2),
        }
        return label, conf, scores

    # softmax (2 outputs assumed [real, fake])
    probs = y[0].astype(float)
    idx = int(np.argmax(probs))
    conf = round(float(probs[idx]) * 100, 2)

    class_names = ["REAL", "FAKE (AI-Generated)"]
    label = class_names[idx] if idx < len(class_names) else "UNKNOWN"

    scores = {
        "real": round(float(probs[0]) * 100, 2) if len(probs) > 0 else 0,
        "fake": round(float(probs[1]) * 100, 2) if len(probs) > 1 else 0,
    }
    return label, conf, scores


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        # chat
        result=None, confidence=None, keywords=[], prob_map={}, explain=None, chat="",
        # image
        img_result=None, img_confidence=None, img_scores=None, img_url=None
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
        # chat
        result=result, confidence=confidence, keywords=keywords, prob_map=prob_map, explain=explain, chat=chat_text,
        # image (blank)
        img_result=None, img_confidence=None, img_scores=None, img_url=None
    )


@app.route("/check-image", methods=["POST"])
def check_image():
    file = request.files.get("image")

    if not file or file.filename == "":
        return render_template(
            "index.html",
            result=None, confidence=None, keywords=[], prob_map={}, explain=None, chat="",
            img_result="No file selected", img_confidence=None, img_scores=None, img_url=None
        )

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            result=None, confidence=None, keywords=[], prob_map={}, explain=None, chat="",
            img_result="Invalid file type (jpg/png/jpeg/webp only)", img_confidence=None, img_scores=None, img_url=None
        )

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    label, conf, scores = predict_image(save_path)
    img_url = f"/static/uploads/{filename}"

    return render_template(
        "index.html",
        result=None, confidence=None, keywords=[], prob_map={}, explain=None, chat="",
        img_result=label, img_confidence=conf, img_scores=scores, img_url=img_url
    )


# ✅ sitemap.xml serve
@app.route("/sitemap.xml")
def sitemap():
    return send_from_directory(app.root_path, "sitemap.xml")


# ✅ robots.txt serve
@app.route("/robots.txt")
def robots():
    return send_from_directory(app.root_path, "robots.txt")


# ✅ health check (Render setting /healthz)
@app.route("/healthz")
def healthz():
    return "ok", 200


# -----------------------------
# Main (local run)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
