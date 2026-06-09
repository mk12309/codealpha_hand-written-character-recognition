# Handwritten Character Recognition

This project uses the EMNIST Balanced dataset from Kaggle to recognize handwritten digits and letters.

## What it does

- Trains a multiclass classifier on EMNIST Balanced
- Decodes the EMNIST label mapping to real characters
- Lets you draw your own character on the page
- Sends the drawing to a local prediction API
- Shows the predicted character, confidence, and top-5 results in real time

## Dataset

The dataset archive includes:

- `emnist-balanced-train.csv`
- `emnist-balanced-test.csv`
- `emnist-balanced-mapping.txt`

## Project files

```text
codealpha_hand written character recognition/
  README.md
  TASK3_REPORT.md
  results.html
  requirements.txt
  src/
    __init__.py
    emnist.py
    train.py
    server.py
    predict.py
  archive/
```

## How to run

### 1. Install dependencies

```powershell
cd "C:\code alpha\codealpha_hand written character recognition"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

```powershell
python -m src.train
```

### 3. Start the local web app

```powershell
python -m src.server
```

### 4. Open the page

Go to:

```text
http://localhost:8080
```

## Output

The page shows:

- drawing canvas
- predicted character
- confidence percentage
- top-5 predictions

## Notes

- The page must be opened through `http://localhost:8080` so it can call the prediction API.
- The result updates from the trained EMNIST model, not hardcoded demo text.
