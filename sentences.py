import re


def split_sentences(text):
    # Don't split after titles like "Mr." ("... met Mrs. Brown").
    return re.split(r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<=[.!?])\s+", text.strip())
