from brain.language.sentences import split_sentences


def test_split_sentences_keeps_titles():
    assert split_sentences("He met Mrs. Brown. She smiled!") == ["He met Mrs. Brown.", "She smiled!"]
