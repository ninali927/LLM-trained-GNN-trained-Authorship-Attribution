import spacy

nlp = spacy.load("en_core_web_md")


def annotate_tokens(text):
    """
    Annotate each token with token text, POS, and entity type.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list
        List of tuples (token_text, pos, ent_type).
    """
    doc = nlp(text)
    annotated_tokens = []

    for token in doc:
        pos = token.pos_
        ent_type = token.ent_type_ if token.ent_type_ else ""
        annotated_tokens.append((token.text, pos, ent_type))

    return annotated_tokens


def mask_named_entities(text, name_placeholder="<NAME>"):
    """
    Replace PERSON, GPE, and ORG entities with a placeholder.

    Parameters
    ----------
    text : str
        Input text.
    name_placeholder : str
        Placeholder token.

    Returns
    -------
    str
        Masked text.
    """
    doc = nlp(text)
    masked_tokens = []

    for token in doc:
        if token.ent_type_ in ["PERSON", "GPE", "ORG"]:
            masked_tokens.append(name_placeholder)
        else:
            masked_tokens.append(token.text)

    masked_text = spacy.tokens.Doc(doc.vocab, words=masked_tokens).text
    return masked_text


def annotate_and_mask(text, name_placeholder="<NAME>"):
    """
    Annotate tokens and mask named entities.

    Parameters
    ----------
    text : str
        Input text.
    name_placeholder : str
        Placeholder token.

    Returns
    -------
    tuple
        (annotated_tokens, masked_text)
    """
    annotation = annotate_tokens(text)
    masked_text = mask_named_entities(text, name_placeholder)

    return annotation, masked_text