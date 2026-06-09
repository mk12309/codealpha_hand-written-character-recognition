from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file

from .emnist import (
    MODEL_PATH,
    ROOT_DIR,
    load_bundle,
    preprocess_canvas_data,
    top_predictions,
)
from .train import train_model


app = Flask(__name__)


def load_or_train_bundle() -> dict:
    if MODEL_PATH.exists():
        return load_bundle()
    return train_model()


BUNDLE = load_or_train_bundle()


@app.get("/")
def index():
    return send_file(ROOT_DIR / "results.html")


@app.get("/api/health")
def health():
    metrics = BUNDLE["metrics"]
    return jsonify(
        {
            "status": "ok",
            "dataset": metrics["dataset"],
            "accuracy": metrics["accuracy"],
            "top5_accuracy": metrics["top5_accuracy"],
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "Missing image data."}), 400

    x = preprocess_canvas_data(image_data)
    probabilities = BUNDLE["model"].predict_proba(x)[0]
    label_index = int(np.argmax(probabilities))
    label_map = BUNDLE["label_map"]

    top5 = top_predictions(probabilities, label_map, limit=5)
    return jsonify(
        {
            "predicted_label": label_index,
            "predicted_character": label_map[label_index],
            "confidence": round(float(probabilities[label_index]) * 100, 2),
            "top_predictions": [
                {
                    "label": prediction.label,
                    "character": prediction.character,
                    "confidence": round(prediction.probability * 100, 2),
                }
                for prediction in top5
            ],
            "metrics": BUNDLE["metrics"],
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True, use_reloader=False)
