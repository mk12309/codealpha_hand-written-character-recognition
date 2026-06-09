# CodeAlpha Task 3 Report

## Task

**Task 3: Handwritten Character Recognition**

## Objective

Recognize handwritten digits and letters using the EMNIST dataset and present the result in a prediction UI.

## Dataset

The Kaggle archive includes the EMNIST dataset files:

- `emnist-balanced-train.csv`
- `emnist-balanced-test.csv`
- `emnist-balanced-mapping.txt`

Additional EMNIST splits are also available:

- `emnist-byclass`
- `emnist-bymerge`
- `emnist-digits`
- `emnist-letters`
- `emnist-mnist`

## Expected Output

The output should match the handwritten recognition interface:

- a drawing area for user input
- the predicted character shown prominently
- confidence percentage
- top-5 predictions with bars or percentages

## Example Output

- Predicted Character: `V`
- Confidence: `98.76%`
- Top-5 Predictions:
  - `V`
  - `Y`
  - `U`
  - `W`
  - `X`

## How the Output Should Look

The final dashboard should contain:

1. A left panel for drawing input
2. A right panel for the prediction result
3. A top-5 predictions section
4. A run button to trigger recognition
5. A footer or summary line showing model and dataset details

## Conclusion

This task is focused on handwritten character recognition using EMNIST. The project output should be a clean, interactive demo that predicts the character and shows confidence in real time.
