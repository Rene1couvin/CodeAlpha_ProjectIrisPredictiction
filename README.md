# Iris Flower Classification

## Project Overview
This project uses machine learning to classify Iris flowers into three species: Setosa, Versicolor, and Virginica, based on their sepal and petal measurements.

## Objectives
- Explore and prepare the Iris dataset.
- Train a classification model using Scikit-learn.
- Evaluate performance using test data.
- Predict the species of a new flower.

## Dataset
The dataset contains 150 records, with 50 samples for each species.

Input features:
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm

Target: `Species`

The `Id` column is excluded from training because it is only a record identifier.

## Technologies
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Model
The project uses Logistic Regression with StandardScaler in a Scikit-learn pipeline.

The dataset is divided into 80% training data and 20% testing data.

## Results
- Test accuracy: 93.33%
- Correct predictions: 28 out of 30 test samples
- Setosa precision and recall: 1.00
- Versicolor precision and recall: 0.90
- Virginica precision and recall: 0.90

These results correspond to the current train/test split and random state.

## How to Run

1. Install Python.
2. Create and activate a virtual environment.
3. Install the required libraries:

   `pip install -r requirements.txt`

4. Place `Iris.csv` in the `data` folder.
5. Train and evaluate the model:

   `python main.py`

6. Predict a new flower:

   `python predict.py`

## Project Structure

- `data/` — Iris dataset
- `models/` — Saved trained model
- `main.py` — Training and evaluation
- `predict.py` — Prediction using the saved model
- `requirements.txt` — Project dependencies
- `confusion_matrix.png` — Evaluation visualization

## Author
CodeAlpha Data Science Internship Project
