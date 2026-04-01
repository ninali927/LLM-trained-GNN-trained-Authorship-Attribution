from preprocess.preprocess_pipeline import preprocess_chunk_text
from WAN.wan_matrix import build_wan_from_sentences
from WAN.markov_normalization import markov_normalization
from WAN.relative_entropy.Kullback_Leibler_Divergence import Kullback_Leibler_Divergence
from WAN.relative_entropy.Jensen_Shannon_Divergence import Jensen_Shannon_Divergence
from WAN.relative_entropy.Hellinger_Distance import Hellinger_Distance
from WAN.relative_entropy.Total_Variation_Distance import Total_Variation_Distance
from WAN.relative_entropy.Bhattacharyya_Distance import Bhattacharyya_Distance
from WAN.relative_entropy.Renyi_Divergence import Renyi_Divergence


def WAN_distance_pipeline(chunk_text_1, chunk_text_2,
                          function_words,
                          D=10, alpha=0.75,
                          epsilon=1e-12,
                          distance_type="kl"):
    """
    Full WAN pipeline between two chunks.

    chunk1
        ↓
    preprocess
        ↓
    WAN matrix
        ↓
    Markov normalization
        ↓
    distance
    """

    # ---- chunk 1 ----
    _, _, _, sentences1 = preprocess_chunk_text(chunk_text_1)

    A1 = build_wan_from_sentences(
        sentences1,
        function_words=function_words,
        D=D,
        alpha=alpha
    )

    P1 = markov_normalization(A1)

    # ---- chunk 2 ----
    _, _, _, sentences2 = preprocess_chunk_text(chunk_text_2)

    A2 = build_wan_from_sentences(
        sentences2,
        function_words=function_words,
        D=D,
        alpha=alpha
    )

    P2 = markov_normalization(A2)

    # ---- distance ----
    if distance_type == "kl":
        return Kullback_Leibler_Divergence(P1, P2, epsilon)

    elif distance_type == "jsd":
        return Jensen_Shannon_Divergence(P1, P2)

    elif distance_type == "hellinger":
        return Hellinger_Distance(P1, P2)

    elif distance_type == "tv":
        return Total_Variation_Distance(P1, P2)

    elif distance_type == "bhattacharyya":
        return Bhattacharyya_Distance(P1, P2)

    elif distance_type == "renyi":
        return Renyi_Divergence(P1, P2)
