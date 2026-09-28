import numpy as np


def safety_gated_predict(proba, classes, tau):
    """Cost-sensitive decision rule: a candidate is labelled Protective only if the model is
    confident (P(Protective) >= tau). Otherwise it falls back to the most likely non-Protective
    class. A false rejection only delays escalation; a false-safe could expose a protected zone."""
    classes = np.asarray(classes)
    pi = int(np.where(classes == "Protective")[0][0])
    masked = proba.copy(); masked[:, pi] = -1
    fallback = classes[masked.argmax(1)]
    return np.where(proba[:, pi] >= tau, "Protective", fallback)
