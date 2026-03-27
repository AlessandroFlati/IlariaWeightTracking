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

def _fit_exponential_decay(days, weights):
    def model(t, a, b, c):
        return a * np.exp(-b * t) + c

    try:
        p0 = [weights[0] - weights[-1], 0.01, weights[-1]]
        popt, _ = curve_fit(model, days, weights, p0=p0, maxfev=10000)
        predicted = model(days, *popt)
        r2 = compute_r_squared(weights, predicted)
        std = np.std(weights - predicted)
        return {"predict": lambda t, p=popt: model(t, *p), "r_squared": r2, "residual_std": std}
    except RuntimeError:
        return None

def _fit_hall_linearized(days, weights):
    def model(t, w_final, w0, tau):
        return w_final + (w0 - w_final) * np.exp(-t / tau)

    try:
        p0 = [weights[-1] - 5, weights[0], 200.0]
        popt, _ = curve_fit(model, days, weights, p0=p0, maxfev=10000)
        predicted = model(days, *popt)
        r2 = compute_r_squared(weights, predicted)
        std = np.std(weights - predicted)
        return {"predict": lambda t, p=popt: model(t, *p), "r_squared": r2, "residual_std": std}
    except RuntimeError:
        return None

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
    results["Polynomial (degree 2)"] = _fit_polynomial(days, weights, 2)
    results["Polynomial (degree 3)"] = _fit_polynomial(days, weights, 3)

    exp_result = _fit_exponential_decay(days, weights)
    if exp_result:
        results["Exponential Decay"] = exp_result

    hall_result = _fit_hall_linearized(days, weights)
    if hall_result:
        results["Hall Linearized (2011)"] = hall_result

    thomas_result = _fit_thomas(days, weights)
    if thomas_result:
        results["Thomas (2013)"] = thomas_result

    return results
