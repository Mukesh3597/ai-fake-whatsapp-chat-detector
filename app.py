import os
import math
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import joblib

import numpy as np
from PIL import Image, ImageStat

app = Flask(_name_)

# -----------------------------
# Paths / Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(_file_))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------
# Load Chat Model (ONCE)
# -----------------------------
CHAT_MODEL_PATH = os.path.join(MODELS_DIR, "chat_detector_model.pkl")
chat_model = joblib.load(CHAT_MODEL_PATH)


# -----------------------------
# Chat Prediction (same as before)
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
# Image Prediction (DEPLOY-SAFE heuristic)
# -----------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict_image(file_path: str):
    """
    Heuristic demo detector:
    - AI images often have smoother textures, different noise/edge stats etc.
    - We'll compute a "fake score" from variance/contrast/entropy-ish signals.
    This is NOT perfect, but gives stable output and deploys 100% on Render.
    """
    img = Image.open(file_path).convert("RGB")
    img_small = img.resize((256, 256))

    # basic stats
    stat = ImageStat.Stat(img_small)
    mean = np.array(stat.mean)          # [R,G,B]
    stdv = np.array(stat.stddev)        # [R,G,B]
    contrast = float(stdv.mean())

    # grayscale variance (texture)
    gray = img_small.convert("L")
    g = np.asarray(gray, dtype=np.float32) / 255.0
    var = float(g.var())
    mean_g = float(g.mean())

    # edge-like measure (simple gradient)
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    grad = float((gx + gy) / 2.0)

    # a combined score (tuned to give nice confidence numbers)
    # lower texture + low gradients often => "more fake-like"
    raw = (
        (0.60 - var) * 4.0 +
        (0.08 - grad) * 10.0 +
        (0.18 - (contrast / 255.0)) * 6.0 +
        (abs(0.50 - mean_g)) * 1.5
    )

    prob_fake = float(np.clip(_sigmoid(raw), 0.01, 0.99))

    label = "FAKE (AI-Generated)" if prob_fake >= 0.5 else "REAL"
    confidence = round((prob_fake if prob_fake >= 0.5 else (1 - prob_fake)) * 100, 2)

    scores = {
        "real": round((1 - prob_fake) * 100, 2),
        "fake": round(prob_fake * 100, 2),
    }
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
        # chat
        result=result, confidence=confidence, keywords=keywords, prob_map=prob_map, explain=explain, chat=chat_text,
        # image
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


if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=False)


from flask import send_from_directory

@app.route("/sitemap.xml")
def sitemap():
    return send_from_directory(app.root_path, "sitemap.xml")


@app.route("/robots.txt")
def robots():
    return send_from_directory(app.root_path, "robots.txt")
