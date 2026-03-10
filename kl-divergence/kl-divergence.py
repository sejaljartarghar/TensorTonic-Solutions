import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # avoid log(0)
    q = q + eps

    # only compute where p > 0
    mask = p > 0

    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))

    return float(kl)