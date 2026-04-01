import re

def remove_extra_spaces(text):
    """
    Clean spacing while keeping newlines.
    """
    text = text.replace("\xa0", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"^[ ]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ ]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n+", "\n", text, flags=re.MULTILINE)

    return text