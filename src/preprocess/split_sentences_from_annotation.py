STOPPERS = {".", "?", "!", ";"}


def split_sentences_from_annotation(annotation, stoppers=STOPPERS):
    """
    Split token annotation into sentence-like chunks using stopper tokens.

    Parameters
    ----------
    annotation : list
        List of tuples (token_text, pos, ent_type).
    stoppers : set
        Tokens that end a sentence.

    Returns
    -------
    list
        List of sentences, where each sentence is a list of token tuples.
    """
    sentences = []
    current = []

    for tok, pos, ent in annotation:
        if pos == "SPACE":
            continue

        current.append((tok, pos, ent))

        if tok in stoppers:
            sentences.append(current)
            current = []

    if current:
        sentences.append(current)

    return sentences