from preprocess.preprocess_pipeline import preprocess_chunk_text
from WAN.wan_matrix import build_wan_from_sentences
from WAN.markov_normalization import markov_normalization
from WAN.function_words import FUNCTION_WORDS
from WAN.relative_entropy.Kullback_Leibler_Divergence import Kullback_Leibler_Divergence
from WAN.relative_entropy.Jensen_Shannon_Divergence import Jensen_Shannon_Divergence
from WAN.relative_entropy.Hellinger_Distance import Hellinger_Distance
from WAN.relative_entropy.Total_Variation_Distance import Total_Variation_Distance
from WAN.relative_entropy.Bhattacharyya_Distance import Bhattacharyya_Distance
from WAN.relative_entropy.Renyi_Divergence import Renyi_Divergence


def build_WAN_markov_chain(chunk_text, function_words=FUNCTION_WORDS, D=10, alpha=0.75):
    """
    Preprocess one chunk and convert it into a WAN Markov chain.

    """
    cleaned_text, annotation, masked_text, sentences = preprocess_chunk_text(chunk_text)

    A = build_wan_from_sentences(
        sentences,
        function_words=function_words,
        D=D,
        alpha=alpha
    )

    P = markov_normalization(A)

    return cleaned_text, annotation, masked_text, sentences, A, P


def compute_chunk_distance(chunk_text_1, chunk_text_2,
                           function_words=FUNCTION_WORDS,
                           D=10, alpha=0.75, epsilon=1e-12,
                           distance_type="kl", renyi_alpha=0.5):
    """
    Compute WAN-based distance between two chunks.

    """
    _, _, _, _, _, P1 = build_WAN_markov_chain(
        chunk_text_1,
        function_words=function_words,
        D=D,
        alpha=alpha
    )

    _, _, _, _, _, P2 = build_WAN_markov_chain(
        chunk_text_2,
        function_words=function_words,
        D=D,
        alpha=alpha
    )

    if distance_type == "kl":
        return Kullback_Leibler_Divergence(P1, P2, epsilon=epsilon)

    elif distance_type == "jsd":
        return Jensen_Shannon_Divergence(P1, P2, epsilon=epsilon)

    elif distance_type == "hellinger":
        return Hellinger_Distance(P1, P2)

    elif distance_type == "tv":
        return Total_Variation_Distance(P1, P2)

    elif distance_type == "bhattacharyya":
        return Bhattacharyya_Distance(P1, P2, epsilon=epsilon)

    elif distance_type == "renyi":
        return Renyi_Divergence(P1, P2, alpha=renyi_alpha, epsilon=epsilon)