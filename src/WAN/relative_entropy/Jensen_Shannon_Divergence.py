import numpy as np
from WAN.relative_entropy.Kullback_Leibler_Divergence import Kullback_Leibler_Divergence


def Jensen_Shannon_Divergence(P1, P2, epsilon=1e-12):
    """
    JSD(P1 || P2) = 1/2 KL(P1 || M) + 1/2 KL(P2 || M)

    where M = 1/2 (P1 + P2).
    """
    M = 0.5 * (P1 + P2)

    value = 0.5 * Kullback_Leibler_Divergence(P1, M, epsilon)
    value += 0.5 * Kullback_Leibler_Divergence(P2, M, epsilon)

    return value