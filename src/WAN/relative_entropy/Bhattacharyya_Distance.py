import numpy as np
from WAN.markov_normalization import compute_stationary_distribution


def Bhattacharyya_Distance(P1, P2, epsilon=1e-12):
    """
    D_B(P1, P2) = -log( sum_i sqrt(pi1_i * pi2_i) )

    where pi1 and pi2 are the stationary distributions of P1 and P2.
    """
    pi1 = compute_stationary_distribution(P1)
    pi2 = compute_stationary_distribution(P2)

    value = 0.0
    n = len(pi1)

    for i in range(n):
        value += np.sqrt(pi1[i] * pi2[i])

    value = -np.log(max(value, epsilon))

    return value


def get_bhattacharyya_annoy_vector(P):
    """
    Return the Annoy-compatible vector for Bhattacharyya distance.

    D_B(P1, P2) = -log( <sqrt(pi1), sqrt(pi2)> )

    So we map each WAN to:
        v = sqrt(pi)

    Then Annoy can approximate nearest neighbors using inner product / cosine similarity.
    """
    pi = compute_stationary_distribution(P)

    # numerical stability
    pi = np.maximum(pi, 1e-12)

    # elementwise sqrt
    v = np.sqrt(pi)

    return v