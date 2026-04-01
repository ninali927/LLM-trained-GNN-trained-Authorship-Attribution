import numpy as np
from WAN.markov_normalization import compute_stationary_distribution


def Total_Variation_Distance(P1, P2):
    """
    TV(P1, P2) = 1/2 * sum_i |pi1_i - pi2_i|

    where pi1 and pi2 are the stationary distributions of P1 and P2.
    """
    pi1 = compute_stationary_distribution(P1)
    pi2 = compute_stationary_distribution(P2)

    value = 0.0
    n = len(pi1)

    for i in range(n):
        value += abs(pi1[i] - pi2[i])

    value = 0.5 * value

    return value