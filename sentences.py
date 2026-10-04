import re


def split_sentences(text):
    # Don't split after titles like "Mr." ("... met Mrs. Brown").
    return re.split(r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<=[.!?])\s+", text.strip())


def upper_first(text):
    return text[:1].upper() + text[1:]


def lower_first(text):
    return text[:1].lower() + text[1:]
