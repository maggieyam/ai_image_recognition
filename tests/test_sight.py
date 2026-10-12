from brain.perception.sight import Sight, _caption_and_description
from brain.workspace import Workspace


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


def test_sight_posts_the_caption_and_description_and_tells_the_page():
    class FakeFlorence:
        def describe(self, image):
            return f"The image shows a grey cat on a windowsill. It is raining. ({image})"

    workspace = Workspace(image="photo")
    events = list(Sight(FakeFlorence()).run(workspace))
    assert workspace["caption"] == "A grey cat on a windowsill."
    assert workspace["description"] == "The image shows a grey cat on a windowsill. It is raining. (photo)"
    assert events == [{"caption": workspace["caption"], "description": workspace["description"]}]
