# Iris Species Classification with Scikit-learn

This project demonstrates a simple machine learning workflow for classifying Iris species using a decision tree classifier from the Scikit-learn library.

## Task

The goal of this task is to:
1.  Preprocess the Iris Species dataset.
2.  Train a decision tree classifier.
3.  Evaluate the model's performance using accuracy, precision, and recall.

## Dataset

The dataset used is the Iris Species dataset, which is included in the `archive` directory.

## Script

The `iris_classification.py` script performs the following steps:
1.  **Load Data**: Loads the Iris dataset from `archive/Iris.csv`.
2.  **Preprocessing**:
    *   Removes the `Id` column.
    *   Handles missing values (though none are present in this dataset).
    *   Encodes the categorical `Species` column into numerical labels.
3.  **Train-Test Split**: Splits the data into training and testing sets.
4.  **Model Training**: Trains a `DecisionTreeClassifier` on the training data.
5.  **Prediction**: Makes predictions on the test set.
6.  **Evaluation**: Calculates and prints the accuracy, precision, and recall of the model.

## How to Run

1.  Ensure you have Python and the required libraries (`pandas`, `scikit-learn`) installed.
2.  Run the script from your terminal:
    ```bash
    python iris_classification.py
    ```

## Results

The script will output the following metrics:
*   Accuracy
*   Precision
*   Recall

For this particular dataset and model, the expected output is:
```
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
