import io

from PIL import Image

from senses import eye


def as_file(image, fmt="PNG"):
    file = io.BytesIO()
    image.save(file, fmt)
    file.seek(0)
    return file


def test_a_big_picture_is_shrunk_and_any_picture_is_rgb():
    seen = eye.see(as_file(Image.new("L", (4000, 1000))))
    assert seen.mode == "RGB" and seen.size == (eye.MAX_SIDE, eye.MAX_SIDE // 4)
    assert eye.see(as_file(Image.new("RGB", (200, 100)))).size == (200, 100)  # small ones are left alone
