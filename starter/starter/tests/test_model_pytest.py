import numpy as np
import pandas as pd
import pytest
from ..ml.data import process_data
from ..ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split

@pytest.fixture(scope="module")
def dataset():
    # Load the dataset
    data = pd.read_csv("starter/data/census.csv")

    # Preprocess the dataset
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    label = 'salary'

    return process_data(data, categorical_features, label, training=True)

@pytest.fixture(scope="module")
def train_test_split_data(dataset):
    X, y, _, _ = dataset
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture(scope="module")
def train_data(train_test_split_data):
    X_train, _, y_train, _ = train_test_split_data
    return X_train, y_train

@pytest.fixture(scope="module")
def test_data(train_test_split_data):
    _, X_test, _, y_test = train_test_split_data
    return X_test, y_test

@pytest.fixture(scope="module")
def trained_model(train_data):
    X_train, y_train = train_data
    model = train_model(X_train, y_train)
    return model

def test_train_model_output(trained_model):
    """
    Test if train_model returns a model object.
    """
    assert trained_model is not None

def test_inference_output_type(trained_model, test_data):
    """
    Test if inference function returns predictions in the expected format.
    """
    X_test, _ = test_data
    predictions = inference(trained_model, X_test)
    assert isinstance(predictions, np.ndarray)

def test_compute_model_metrics_output(trained_model, test_data):
    """
    Test if compute_model_metrics returns precision, recall, and fbeta in the expected format.
    """
    X_test, y_test = test_data
    preds_test = inference(trained_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds_test)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
