import numpy as np
from scipy.optimize import curve_fit

def compute_r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot

def _fit_polynomial(days, weights, degree):
    coeffs = np.polyfit(days, weights, degree)
    poly = np.poly1d(coeffs)
    predicted = poly(days)
    r2 = compute_r_squared(weights, predicted)
    std = np.std(weights - predicted)
    return {"predict": lambda t, p=poly: p(t), "r_squared": r2, "residual_std": std}

def _fit_thomas(days, weights):
    def model(t, w0, delta_w, k):
        return w0 - delta_w * (1 - np.exp(-k * t))

    try:
        p0 = [weights[0], weights[0] - weights[-1], 0.01]
        popt, _ = curve_fit(model, days, weights, p0=p0, maxfev=10000)
        predicted = model(days, *popt)
        r2 = compute_r_squared(weights, predicted)
        std = np.std(weights - predicted)
        return {"predict": lambda t, p=popt: model(t, *p), "r_squared": r2, "residual_std": std}
    except RuntimeError:
        return None

def fit_all_models(days: np.ndarray, weights: np.ndarray) -> dict:
    results = {}
    results["Linear"] = _fit_polynomial(days, weights, 1)

    thomas_result = _fit_thomas(days, weights)
    if thomas_result:
        results["Thomas (2013)"] = thomas_result

    return results
