from recognizer import MAX_PEOPLE, _caption_and_description, main_characters


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


def face(x, height):
    return {"box": [x, 100, x + 50, 100 + height]}


def test_main_characters_drop_background_faces():
    main, crowd = face(100, 250), face(10, 25)  # 250 and 25 px tall in a 1000 px photo
    assert main_characters([main, crowd], image_height=1000) == [main]


def test_main_characters_keep_the_biggest_faces_left_to_right():
    faces = [face(x, height) for x, height in [(500, 300), (0, 120), (300, 200), (100, 400), (700, 150), (900, 110)]]
    kept = main_characters(faces, image_height=1000)
    assert len(kept) == MAX_PEOPLE
    assert [f["box"][0] for f in kept] == [100, 300, 500, 700]  # the 4 biggest, left to right
