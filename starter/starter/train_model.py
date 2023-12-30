# Script to train machine learning model.
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

import os
print("Current Working Directory:", os.getcwd())

# Load and preprocess data
data = pd.read_csv("data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

# Print out the metrics
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {fbeta}")

# Save the model and encoders
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")
