import numpy as np
from src.models import fit_all_models, compute_r_squared

def test_fit_all_models_returns_dict():
    days = np.array([0, 10, 20, 30, 40, 50], dtype=float)
    weights = np.array([95, 94, 93, 92.5, 92, 91.5], dtype=float)
    results = fit_all_models(days, weights)
    assert isinstance(results, dict)
    expected_names = [
        "Linear",
        "Polynomial (degree 2)",
        "Polynomial (degree 3)",
        "Exponential Decay",
        "Hall Linearized (2011)",
        "Thomas (2013)",
    ]
    for name in expected_names:
        assert name in results, f"Missing model: {name}"
        assert "predict" in results[name]
        assert "r_squared" in results[name]
        assert callable(results[name]["predict"])
        assert 0 <= results[name]["r_squared"] <= 1.0

def test_linear_model_predicts_decrease():
    days = np.array([0, 10, 20, 30, 40, 50], dtype=float)
    weights = np.array([95, 94, 93, 92.5, 92, 91.5], dtype=float)
    results = fit_all_models(days, weights)
    predict = results["Linear"]["predict"]
    assert predict(100) < predict(0)

def test_compute_r_squared():
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    r2 = compute_r_squared(actual, predicted)
    assert 0.95 < r2 <= 1.0

def test_compute_r_squared_perfect():
    actual = np.array([1.0, 2.0, 3.0])
    r2 = compute_r_squared(actual, actual)
    assert r2 == 1.0
