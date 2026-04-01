import numpy as np

def markov_normalization(A):
    """
    Row-normalize WAN adjacency matrix into a Markov transition matrix.
    """
    P = A.copy().astype(float)
    row_sums = P.sum(axis=1, keepdims=True)

    for i in range(P.shape[0]):
        if row_sums[i, 0] > 0:
            P[i, :] = P[i, :] / row_sums[i, 0]

    return P


def compute_stationary_distribution(P, max_iter=10000, tol=1e-12):
    """
    Compute stationary distribution pi for a Markov chain P
    using power iteration.

    Parameters
    ----------
    P : numpy.ndarray
        Row-stochastic matrix.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    numpy.ndarray
        Stationary distribution vector.
    """
    n = P.shape[0]
    pi = np.ones(n) / n

    for _ in range(max_iter):
        new_pi = pi @ P

        if np.linalg.norm(new_pi - pi, ord=1) < tol:
            return new_pi

        pi = new_pi

    return pi