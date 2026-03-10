import numpy as np

def linear_regression_closed_form(X, y):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    XT = X.T
    XTX = XT @ X
    XTy = XT @ y

    w = np.linalg.inv(XTX) @ XTy

    return w