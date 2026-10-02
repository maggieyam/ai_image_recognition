from recognizer import _caption_and_description


def test_drops_film_still_and_title_sentences():
    caption, description = _caption_and_description(
        'The image is a still from the drama "The King\'s Woman". The image shows a woman '
        "in a red dress looking at the camera. She stands in a palace hall."
    )
    assert description == (
        "The image shows a woman in a red dress looking straight ahead. "
        "She stands in a palace hall."
    )
    assert caption == "A woman in a red dress looking straight ahead."


def test_keeps_text_when_every_sentence_names_a_title():
    caption, description = _caption_and_description('A poster for the film "Moon River".')
    assert description == "A poster for the film."
    assert caption == "A poster for the film."
