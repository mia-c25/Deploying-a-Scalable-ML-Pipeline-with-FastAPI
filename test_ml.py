import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)


@pytest.fixture
def sample_data():
    """Provide a small binary-classification dataset for model testing."""
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_train_model_returns_random_forest(sample_data):
    """Confirm that training returns a fitted random forest model."""
    X, y = sample_data
    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")


def test_inference_returns_predictions(sample_data):
    """Confirm that inference returns one binary prediction per row."""
    X, y = sample_data
    model = train_model(X, y)
    predictions = inference(model, X)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == y.shape
    assert set(predictions).issubset({0, 1})


def test_compute_model_metrics():
    """Confirm precision, recall, and F1 are calculated correctly."""
    y = np.array([1, 1, 0, 0])
    predictions = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, predictions)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)


def test_save_and_load_model(sample_data, tmp_path):
    """Confirm that a saved model can be loaded and used."""
    X, y = sample_data
    model = train_model(X, y)
    model_path = tmp_path / "test_model.pkl"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    original_predictions = inference(model, X)
    loaded_predictions = inference(loaded_model, X)

    assert np.array_equal(original_predictions, loaded_predictions)
