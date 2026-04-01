import numpy as np
from WAN.function_words import FUNCTION_WORDS


def build_wan_from_sentences(sentences, function_words=FUNCTION_WORDS, D=10, alpha=0.75):
    """
    Build WAN adjacency matrix from sentence annotations.

    """

    F = sorted(list(function_words))
    idx = {w: i for i, w in enumerate(F)}

    n = len(F)

    A = np.zeros((n, n), dtype=float)

    for sentence in sentences:

        L = len(sentence)

        for e in range(L):

            wi = sentence[e][0].lower()
            if wi not in F:
                continue

            i = idx[wi]

            max_d = min(D, L - e - 1)

            for d in range(1, max_d + 1):

                wj = sentence[e + d][0].lower()

                if wj in F:
                    j = idx[wj]
                    A[i, j] += alpha ** (d - 1)

    return A