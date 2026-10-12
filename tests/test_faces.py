from PIL import Image

from brain.perception.faces import MAX_PEOPLE, ReadFaces, _clean_boxes, feeling, main_characters
from brain.workspace import Workspace


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


class FakeFinder:
    """Finds the given face boxes and captions every crop the same; records how often it looked."""

    def __init__(self, boxes):
        self.boxes, self.calls = boxes, 0

    def find_faces(self, image):
        self.calls += 1
        return self.boxes

    def caption(self, image):
        return "A man with a beard."


def test_florence_looks_for_a_close_up_yunet_missed():
    yunet, florence = FakeFinder([[0, 0, 10, 10]]), FakeFinder([[100, 100, 400, 450]])  # only a tiny face in the background
    found = ReadFaces(yunet, florence, None).find_characters(Image.new("RGB", (500, 500)))
    assert [c["box"] for c in found] == [[100, 100, 400, 450]]
    assert found[0]["character"] == "a man with a beard" and found[0]["face"].size == (300, 350)


def test_a_crowd_of_small_faces_doesnt_make_florence_look():
    crowd = [[x, 10, x + 10, 20] for x in range(0, 100, 20)]
    yunet, florence = FakeFinder(crowd), FakeFinder([])
    assert ReadFaces(yunet, florence, None).find_characters(Image.new("RGB", (500, 500))) == []
    assert florence.calls == 0


class FakeEmotions:
    """Reads the emotion written on each fake face, one face per call."""

    def __init__(self):
        self.faces = []

    def recognize(self, face):
        self.faces.append(face)
        return {"emotion": face, "valence": 0.0, "arousal": 0.0}


MAN = {"character": "a man with a beard", "face": "Anger"}
GIRL = {"character": "a girl crying", "face": "Sadness"}


def read(characters, emotions=None):
    """What reading these characters' faces posts and tells the page, and the fake models."""
    emotions = emotions or FakeEmotions()
    faces, looked = ReadFaces(None, None, emotions), []
    faces.find_characters = lambda image: looked.append(image) or [dict(c) for c in characters]
    workspace = Workspace(image="image")
    events = list(faces.run(workspace))
    return workspace["characters"], events, emotions, looked


def test_each_characters_face_is_read_and_posted_without_the_face():
    characters, events, emotions, _ = read([MAN, GIRL])
    assert emotions.faces == ["Anger", "Sadness"]  # one face at a time
    assert [c["emotion"] for c in characters] == ["Anger", "Sadness"]
    assert all("face" not in c for c in characters)
    assert events == [{"status": "Looking at their faces…", "for": "feelings"}, {"characters": characters}]


def test_an_animals_feeling_is_recognized():
    dog = {"character": "a brown dog on a red couch", "face": "Fear"}
    assert [c["emotion"] for c in read([dog])[0]] == ["Fear"]


def test_faces_are_always_looked_for_and_none_found_is_posted_too():
    characters, events, emotions, looked = read([])
    assert looked == ["image"]
    assert characters == [] and events[1] == {"characters": []}
    assert emotions.faces == []


def test_a_face_that_cant_be_read_leaves_out_only_that_character():
    class OneBadFace(FakeEmotions):
        def recognize(self, face):
            if face == "Anger":
                raise ValueError("empty crop")
            return super().recognize(face)

    characters, _, _, _ = read([MAN, GIRL], emotions=OneBadFace())
    assert [c["character"] for c in characters] == ["a girl crying"]


def test_no_characters_when_looking_for_faces_fails():
    class Broken:
        def find_faces(self, image):
            raise RuntimeError("detector crashed")

    workspace = Workspace(image=Image.new("RGB", (500, 500)))
    events = list(ReadFaces(Broken(), None, None).run(workspace))
    assert workspace["characters"] == [] and events[1] == {"characters": []}


def test_an_emotion_label_as_a_feeling():
    assert feeling("Sadness") == "sad" and feeling("Awe") == "awe"
