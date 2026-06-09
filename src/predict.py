from __future__ import annotations

import argparse
import json
from pathlib import Path

from .emnist import MODEL_PATH, load_bundle, preprocess_canvas_data, top_predictions
from .train import train_model


def load_model() -> dict:
    if MODEL_PATH.exists():
        return load_bundle()
    return train_model()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict an EMNIST character from a PNG or data URL.")
    parser.add_argument("--image", required=True, help="Path to an image file or a data URL string.")
    args = parser.parse_args()

    bundle = load_model()
    label_map = bundle["label_map"]

    image_arg = args.image
    if Path(image_arg).exists():
        with open(image_arg, "rb") as handle:
            import base64

            image_arg = "data:image/png;base64," + base64.b64encode(handle.read()).decode("utf-8")

    x = preprocess_canvas_data(image_arg)
    probabilities = bundle["model"].predict_proba(x)[0]
    label_index = int(probabilities.argmax())
    top5 = top_predictions(probabilities, label_map, limit=5)

    print(
        json.dumps(
            {
                "predicted_character": label_map[label_index],
                "confidence": round(float(probabilities[label_index]) * 100, 2),
                "top_predictions": [
                    {
                        "character": item.character,
                        "confidence": round(item.probability * 100, 2),
                    }
                    for item in top5
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
