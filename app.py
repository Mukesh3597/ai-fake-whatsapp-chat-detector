import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import joblib

import numpy as np
from PIL import Image
import tensorflow as tf

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


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------
# Load Models (safe paths)
# -----------------------------
CHAT_MODEL_PATH = os.path.join(MODELS_DIR, "chat_detector_model.pkl")
IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "ai_image_detector.h5")

chat_model = joblib.load(CHAT_MODEL_PATH)
image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)

IMG_SIZE = (224, 224)


# -----------------------------
# Chat Prediction
# -----------------------------
def predict_with_details(text: str):
    pred = chat_model.predict([text])[0]

    proba = chat_model.predict_proba([text])[0]
    classes = list(chat_model.classes_)
    prob_map = {classes[i]: round(float(proba[i]) * 100, 2) for i in range(len(classes))}
    confidence = prob_map.get(pred, round(float(proba.max()) * 100, 2))

    if pred == "fake":
        explain = "Scam/Threat pattern detected (OTP/KYC/block/click/refund type words)."
    elif pred == "suspicious":
        explain = "Forward/Chain style detected (share/forward/urgent/breaking type words)."
    else:
        explain = "Looks like normal human conversation."

    # Top keywords from TF-IDF
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
        if len(w) < 3:
            continue
        keywords.append(w)
        if len(keywords) == 5:
            break

    return pred, confidence, keywords, prob_map, explain


# -----------------------------
# Image Prediction
# -----------------------------
def predict_image(file_path: str):
    img = Image.open(file_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    y = image_model.predict(x, verbose=0)

    # sigmoid output (1)
    if y.shape[-1] == 1:
        prob_fake = float(y[0][0])
        label = "FAKE (AI-Generated)" if prob_fake >= 0.5 else "REAL"
        confidence = round((prob_fake if prob_fake >= 0.5 else (1 - prob_fake)) * 100, 2)
        scores = {"real": round((1 - prob_fake) * 100, 2), "fake": round(prob_fake * 100, 2)}
        return label, confidence, scores

    # softmax output (2)
    probs = y[0].astype(float)
    idx = int(np.argmax(probs))
    confidence = round(float(probs[idx]) * 100, 2)

    # default mapping: [real, fake]
    class_names = ["REAL", "FAKE (AI-Generated)"]
    label = class_names[idx]
    scores = {"real": round(float(probs[0]) * 100, 2), "fake": round(float(probs[1]) * 100, 2)}
    return label, confidence, scores


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
        result=result, confidence=confidence, keywords=keywords, prob_map=prob_map, explain=explain, chat=chat_text,
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
