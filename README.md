# Handwritten Character Recognition

This project uses the EMNIST dataset from Kaggle to recognize handwritten digits and letters.

## Task

Build a handwritten character recognition system that classifies an input drawing as a digit `0-9` or a letter `A-Z`.

## Dataset

The dataset archive contains EMNIST variants, including:

- `emnist-balanced`
- `emnist-byclass`
- `emnist-bymerge`
- `emnist-digits`
- `emnist-letters`
- `emnist-mnist`

For a balanced digit-and-letter demo, the best fit is:

- `emnist-balanced-train.csv`
- `emnist-balanced-test.csv`
- `emnist-balanced-mapping.txt`

## Expected Output

The application should display:

- a drawing canvas
- the predicted character
- confidence percentage
- top-5 predictions with probabilities

## Example Result

- Predicted Character: `V`
- Confidence: `98.76%`
- Top-5 Predictions: `V, Y, U, W, X`

## Suggested Workflow

1. Load the EMNIST Balanced dataset
2. Decode numeric labels using the mapping file
3. Normalize pixel values
4. Train a multi-class classifier
5. Show prediction and confidence in a clean UI

## Suggested Submission Text

This project demonstrates handwritten character recognition using the EMNIST Balanced dataset. The model accepts a drawn character, predicts the most likely class, and displays the confidence score with the top-5 class probabilities.
