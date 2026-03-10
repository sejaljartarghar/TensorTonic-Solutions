import numpy as np

def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=float)

    N = y_true.shape[0]

    # select probabilities of the correct classes
    p = y_pred[np.arange(N), y_true]

    # compute average cross-entropy loss
    loss = -np.mean(np.log(p))

    return float(loss)