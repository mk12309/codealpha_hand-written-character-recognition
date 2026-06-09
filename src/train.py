from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score

from .emnist import (
    DATASET_TEST,
    DATASET_TRAIN,
    METRICS_PATH,
    load_label_map,
    orient_emnist_images,
    preprocess_training_batch,
    read_csv_chunks,
    save_bundle,
    top_predictions,
)


def train_model(epochs: int = 6, batch_size: int = 8192) -> dict:
    if not DATASET_TRAIN.exists():
        raise FileNotFoundError(f"Training file not found: {DATASET_TRAIN}")
    if not DATASET_TEST.exists():
        raise FileNotFoundError(f"Test file not found: {DATASET_TEST}")

    label_map = load_label_map()
    classes = np.array(sorted(label_map.keys()), dtype=np.int64)

    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        penalty="l2",
        average=True,
        fit_intercept=True,
        learning_rate="optimal",
        random_state=42,
        tol=None,
    )

    for epoch in range(epochs):
        for batch in read_csv_chunks(DATASET_TRAIN, chunksize=batch_size):
            batch = batch.sample(frac=1.0, random_state=42 + epoch).reset_index(drop=True)
            x_batch, y_batch = preprocess_training_batch(batch)
            classifier.partial_fit(x_batch, y_batch, classes=classes)

    test_frame = pd.read_csv(DATASET_TEST, header=None)
    x_test_raw = test_frame.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0
    x_test = orient_emnist_images(x_test_raw)
    y_test = test_frame.iloc[:, 0].to_numpy(dtype=np.int64)

    probabilities = classifier.predict_proba(x_test)
    predictions = probabilities.argmax(axis=1)

    accuracy = float(accuracy_score(y_test, predictions))
    top5_accuracy = float(top_k_accuracy_score(y_test, probabilities, k=5, labels=classes))
    loss = float(log_loss(y_test, probabilities, labels=classes))

    bundle = {
        "model": classifier,
        "label_map": label_map,
        "classes": classes,
        "metrics": {
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "log_loss": loss,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "epochs": epochs,
            "batch_size": batch_size,
            "dataset": "EMNIST Balanced",
        },
    }

    save_bundle(bundle)

    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(bundle["metrics"], handle, indent=2)

    return bundle


def main() -> None:
    bundle = train_model()
    metrics = bundle["metrics"]
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
