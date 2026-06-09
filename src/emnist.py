from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT_DIR / "archive"
DATASET_TRAIN = ARCHIVE_DIR / "emnist-balanced-train.csv"
DATASET_TEST = ARCHIVE_DIR / "emnist-balanced-test.csv"
MAPPING_FILE = ARCHIVE_DIR / "emnist-balanced-mapping.txt"

ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "emnist_balanced_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "emnist_balanced_metrics.json"


@dataclass(frozen=True)
class Prediction:
    label: int
    character: str
    probability: float


def load_label_map() -> dict[int, str]:
    mapping = pd.read_csv(MAPPING_FILE, sep=r"\s+", header=None, names=["label", "codepoint"])
    return {int(row.label): chr(int(row.codepoint)) for row in mapping.itertuples(index=False)}


def read_csv_chunks(path: Path, chunksize: int):
    return pd.read_csv(path, header=None, chunksize=chunksize)


def split_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = frame.iloc[:, 0].to_numpy(dtype=np.int64)
    x = frame.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0
    return x, y


def orient_emnist_images(x: np.ndarray) -> np.ndarray:
    images = x.reshape(-1, 28, 28)
    images = np.transpose(images, (0, 2, 1))
    images = np.flip(images, axis=2)
    return images.reshape(-1, 784)


def preprocess_training_batch(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x, y = split_features(frame)
    x = orient_emnist_images(x)
    return x, y


def preprocess_canvas_data(image_data: str) -> np.ndarray:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    raw = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(raw)).convert("L")

    image = ImageOps.invert(image)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    if image.width == 0 or image.height == 0:
        image = Image.new("L", (20, 20), 0)
    else:
        scale = 20 / max(image.width, image.height)
        resized = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        image = image.resize(resized, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    left = (28 - image.width) // 2
    top = (28 - image.height) // 2
    canvas.paste(image, (left, top))

    x = np.asarray(canvas, dtype=np.float32) / 255.0
    x = orient_emnist_images(x.reshape(1, 784))
    return x


def top_predictions(probabilities: np.ndarray, label_map: dict[int, str], limit: int = 5) -> list[Prediction]:
    indices = np.argsort(probabilities)[::-1][:limit]
    return [
        Prediction(
            label=int(index),
            character=label_map[int(index)],
            probability=float(probabilities[index]),
        )
        for index in indices
    ]


def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def save_bundle(bundle: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
