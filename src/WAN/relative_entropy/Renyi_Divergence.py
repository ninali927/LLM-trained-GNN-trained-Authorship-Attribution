import numpy as np
from WAN.markov_normalization import compute_stationary_distribution


def Renyi_Divergence(P1, P2, alpha=0.5, epsilon=1e-12):
    """
    D_alpha(P1 || P2) = (1 / (alpha - 1)) * log( sum_i pi1_i^alpha * pi2_i^(1-alpha) )

    where pi1 and pi2 are the stationary distributions of P1 and P2.
    alpha > 0 and alpha != 1.
    """
    if alpha <= 0 or alpha == 1:
        raise ValueError("alpha must be > 0 and != 1")

    pi1 = compute_stationary_distribution(P1)
    pi2 = compute_stationary_distribution(P2)

    value = 0.0
    n = len(pi1)

    for i in range(n):
        p = max(pi1[i], epsilon)
        q = max(pi2[i], epsilon)

        value += (p ** alpha) * (q ** (1 - alpha))

    value = (1 / (alpha - 1)) * np.log(value)

    return value




def get_renyi_annoy_vector(P, alpha=0.5, role="query", epsilon=1e-12):
    """
    Return the Annoy-compatible vector for Rényi divergence.

    Based on the rewrite:
        D_alpha(P1 || P2) ~ log( <P1^alpha, P2^(1-alpha)> )

    So:
        - query vector = pi^alpha
        - index vector = pi^(1-alpha)

    role:
        "query" -> return pi^alpha
        "index" -> return pi^(1-alpha)
    """
    if alpha <= 0 or alpha == 1:
        raise ValueError("alpha must be > 0 and != 1")

    pi = compute_stationary_distribution(P)
    pi = np.maximum(pi, epsilon)

    if role == "query":
        return pi ** alpha
    elif role == "index":
        return pi ** (1 - alpha)
    else:
        raise ValueError("role must be 'query' or 'index'")