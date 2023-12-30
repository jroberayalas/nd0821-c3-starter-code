# Script to train machine learning model.
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

def evaluate_model_on_slices(model, original_data, encoded_data, y, categorical_features):
    with open('model/slice_output.txt', 'a') as f:
        for feature in categorical_features:
            unique_values = original_data[feature].unique()
            for value in unique_values:
                # Get indices where the feature has the specific value
                indices = original_data[original_data[feature] == value].index

                # Filter the one-hot encoded data and labels using these indices
                X_slice = encoded_data[indices]
                y_slice = y[indices]

                # Make predictions on the slice
                preds_slice = model.predict(X_slice)

                # Compute performance metrics
                precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
                
                # Output the results to the file
                f.write(f"Metrics for {feature} = {value}:\n")
                f.write(f"Precision: {precision}, Recall: {recall}, F1 Score: {fbeta}\n\n")

# Load and preprocess data
data = pd.read_csv("data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
test = test.reset_index(drop=True)

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
with open('model/slice_output.txt', 'w') as f:
    f.write(f"Overall Metrics:\n")
    f.write(f"Precision: {precision}, Recall: {recall}, F1 Score: {fbeta}\n\n")

# Evaluate the model on slices of the data
evaluate_model_on_slices(model, test, X_test, y_test, cat_features)

# Save the model and encoders
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")
