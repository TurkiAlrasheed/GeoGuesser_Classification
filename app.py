"""
SoCalGuessr – A GeoGuessr-style web game.

Players see a street-view image and guess which Southern California city
it's from. The trained EfficientNet-B0 model also makes a prediction,
and scores are compared head-to-head.

Usage:
    python app.py
"""

import os
import pathlib
import random
import secrets

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import (
    Flask, render_template, request, session, redirect, url_for, send_file, jsonify
)

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model.pt"

CLASSES = sorted([
    "Anaheim", "Bakersfield", "Los_Angeles",
    "Riverside", "SLO", "San_Diego",
])

DISPLAY_NAMES = {
    "Anaheim": "Anaheim",
    "Bakersfield": "Bakersfield",
    "Los_Angeles": "Los Angeles",
    "Riverside": "Riverside",
    "SLO": "San Luis Obispo",
    "San_Diego": "San Diego",
}

TOTAL_ROUNDS = 10
POINTS_PER_CORRECT = 1000

# ── Model ────────────────────────────────────────────────────────────────────

class FineTunedModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def load_model():
    device = torch.device("cpu")
    model = FineTunedModel(num_classes=len(CLASSES))
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model, device


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_image(model, device, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
    predicted_idx = probs.argmax().item()
    confidence = {
        CLASSES[i]: round(probs[i].item() * 100, 1)
        for i in range(len(CLASSES))
    }
    return CLASSES[predicted_idx], confidence


# ── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

model, device = load_model()

all_images = sorted([f.name for f in DATA_DIR.glob("*.jpg")])
if not all_images:
    raise RuntimeError(
        f"No .jpg images found in {DATA_DIR}. "
        "Place your training images there before starting the app."
    )


def get_ground_truth(filename):
    return filename.rsplit("-", 1)[0]


def pick_round_images():
    """Select TOTAL_ROUNDS random images for a new game, no repeats."""
    count = min(TOTAL_ROUNDS, len(all_images))
    return random.sample(all_images, count)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    images = pick_round_images()
    session["images"] = images
    session["round"] = 0
    session["player_score"] = 0
    session["ai_score"] = 0
    session["history"] = []
    return redirect(url_for("play"))


@app.route("/play")
def play():
    if "images" not in session:
        return redirect(url_for("index"))

    round_num = session.get("round", 0)
    images = session["images"]

    if round_num >= len(images):
        return redirect(url_for("scoreboard"))

    current_image = images[round_num]
    return render_template(
        "game.html",
        image_file=current_image,
        round_num=round_num + 1,
        total_rounds=len(images),
        player_score=session.get("player_score", 0),
        ai_score=session.get("ai_score", 0),
        classes=CLASSES,
        display_names=DISPLAY_NAMES,
    )


@app.route("/guess", methods=["POST"])
def guess():
    if "images" not in session:
        return redirect(url_for("index"))

    player_guess = request.form.get("guess", "")
    round_num = session.get("round", 0)
    images = session["images"]

    if round_num >= len(images):
        return redirect(url_for("scoreboard"))

    current_image = images[round_num]
    ground_truth = get_ground_truth(current_image)

    ai_guess, confidence = predict_image(model, device, DATA_DIR / current_image)

    player_correct = player_guess == ground_truth
    ai_correct = ai_guess == ground_truth

    player_points = POINTS_PER_CORRECT if player_correct else 0
    ai_points = POINTS_PER_CORRECT if ai_correct else 0

    session["player_score"] = session.get("player_score", 0) + player_points
    session["ai_score"] = session.get("ai_score", 0) + ai_points
    session["round"] = round_num + 1

    history = session.get("history", [])
    history.append({
        "image": current_image,
        "ground_truth": ground_truth,
        "player_guess": player_guess,
        "ai_guess": ai_guess,
        "player_correct": player_correct,
        "ai_correct": ai_correct,
    })
    session["history"] = history

    is_last = (round_num + 1) >= len(images)

    sorted_confidence = sorted(confidence.items(), key=lambda x: x[1], reverse=True)

    return render_template(
        "result.html",
        image_file=current_image,
        round_num=round_num + 1,
        total_rounds=len(images),
        player_score=session["player_score"],
        ai_score=session["ai_score"],
        player_guess=player_guess,
        ai_guess=ai_guess,
        ground_truth=ground_truth,
        player_correct=player_correct,
        ai_correct=ai_correct,
        player_points=player_points,
        ai_points=ai_points,
        confidence=sorted_confidence,
        display_names=DISPLAY_NAMES,
        is_last=is_last,
    )


@app.route("/scoreboard")
def scoreboard():
    if "images" not in session:
        return redirect(url_for("index"))

    player_score = session.get("player_score", 0)
    ai_score = session.get("ai_score", 0)
    history = session.get("history", [])
    total_rounds = len(session.get("images", []))

    player_correct_count = sum(1 for h in history if h["player_correct"])
    ai_correct_count = sum(1 for h in history if h["ai_correct"])

    return render_template(
        "scoreboard.html",
        player_score=player_score,
        ai_score=ai_score,
        history=history,
        total_rounds=total_rounds,
        player_correct_count=player_correct_count,
        ai_correct_count=ai_correct_count,
        display_names=DISPLAY_NAMES,
    )


@app.route("/image/<filename>")
def serve_image(filename):
    path = DATA_DIR / filename
    if not path.exists() or not path.is_file():
        return "Not found", 404
    return send_file(path, mimetype="image/jpeg")


if __name__ == "__main__":
    print(f"Loaded {len(all_images)} images from {DATA_DIR}")
    print(f"Model loaded from {MODEL_PATH}")
    print("Starting SoCalGuessr on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
