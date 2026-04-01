FUNCTION_WORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","as","at","by",
    "from","that","this","it","he","she","i","you","we","they","is","was","be","been",
    "are","were","not","do","does","did","have","has","had"
}


def get_function_word_to_idx(function_words=FUNCTION_WORDS):
    """
    Build a dictionary mapping each function word to its row/column index.
    """
    return {word: i for i, word in enumerate(function_words)}