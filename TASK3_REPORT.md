# CodeAlpha Task 3 Report

## Task

**Task 3: Handwritten Character Recognition**

## Objective

Build a handwritten character recognition system using the EMNIST Balanced dataset and display the prediction in a web dashboard.

## Approach

1. Load the EMNIST Balanced CSV files from the Kaggle archive
2. Decode labels using the EMNIST mapping file
3. Normalize and orient the image pixels
4. Train a multiclass classifier
5. Serve predictions through a local Flask API
6. Show the output on the web page with confidence and top-5 probabilities

## Dataset

- Training file: `emnist-balanced-train.csv`
- Test file: `emnist-balanced-test.csv`
- Label map: `emnist-balanced-mapping.txt`

## Output

The page displays:

- drawing canvas for a user-written character
- predicted character
- confidence percentage
- top-5 predictions

## Example Output

- Predicted Character: `V`
- Confidence: `98.76%`
- Top-5 Predictions:
  - `V`
  - `Y`
  - `U`
  - `W`
  - `X`

## How to Run

```powershell
cd "C:\code alpha\codealpha_hand written character recognition"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
python -m src.server
```

## Conclusion

This task connects the handwritten character recognition UI to the EMNIST dataset with a real prediction pipeline, so the page can return live results from the trained model.
