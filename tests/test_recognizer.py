import numpy as np
import pytest
from PIL import Image

import recognizer
from recognizer import MAX_PEOPLE, ImageRecognizer, _caption_and_description, _clean_boxes, main_characters


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


def test_face_boxes_are_kept_inside_the_picture_and_never_empty():
    boxes = [[10.4, 20.6, 60.4, 80.6], [-5.0, 90.0, 25.0, 130.0], [150.0, 20.0, 160.0, 30.0]]
    assert _clean_boxes(boxes, (100, 120)) == [[10, 21, 60, 81], [0, 90, 25, 120]]


def finder(yunet, florence):
    """An ImageRecognizer whose detectors return the given face boxes; records Florence's calls."""
    f = ImageRecognizer.__new__(ImageRecognizer)
    f.florence_calls = 0

    def florence_faces(image):
        f.florence_calls += 1
        return florence

    f._yunet_faces, f._florence_faces = lambda image: yunet, florence_faces
    f._who = lambda image, box: "a man with a beard"
    return f


def test_florence_looks_for_a_close_up_yunet_missed():
    f = finder(yunet=[[0, 0, 10, 10]], florence=[[100, 100, 400, 450]])  # only a tiny face in the background
    assert [c["box"] for c in f.find_characters(Image.new("RGB", (500, 500)))] == [[100, 100, 400, 450]]


def test_a_crowd_of_small_faces_doesnt_make_florence_look():
    crowd = [[x, 10, x + 10, 20] for x in range(0, 100, 20)]
    f = finder(yunet=crowd, florence=[])
    assert f.find_characters(Image.new("RGB", (500, 500))) == [] and f.florence_calls == 0


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def read(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("data", [b"cut off", OSError("connection reset")])
def test_a_bad_yunet_download_is_never_kept(tmp_path, monkeypatch, data):
    monkeypatch.setattr(recognizer, "YUNET_FILE", tmp_path / "yunet.onnx")
    monkeypatch.setattr(recognizer.urllib.request, "urlopen", lambda url, timeout: FakeResponse(data))
    with pytest.raises((RuntimeError, OSError)):
        recognizer._yunet_file()
    assert list(tmp_path.iterdir()) == []  # neither the model nor a leftover temp file
