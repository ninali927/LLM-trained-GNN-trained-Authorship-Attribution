from .remove_extra_spaces import remove_extra_spaces
from .annotate_and_mask import annotate_and_mask
from .split_sentences_from_annotation import split_sentences_from_annotation


def preprocess_chunk_text(text, name_placeholder="<NAME>"):
    """
    Full preprocessing pipeline for one chunk.

    """
    cleaned_text = remove_extra_spaces(text)

    annotation, masked_text = annotate_and_mask(
        cleaned_text,
        name_placeholder=name_placeholder
    )

    sentences = split_sentences_from_annotation(annotation)

    return cleaned_text, annotation, masked_text, sentences