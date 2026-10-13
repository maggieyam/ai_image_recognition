"""The eye: a picture file as a picture the brain can look at."""
from PIL import Image, UnidentifiedImageError

MAX_SIDE = 1600  # downscale big images; the model resizes anyway


class Unreadable(Exception):
    """The file isn't an image."""


def see(file):
    """The picture in `file` (a path or an open file), in RGB and at most MAX_SIDE a side."""
    try:
        image = Image.open(file).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise Unreadable from error
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image
