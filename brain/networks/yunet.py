"""
YuNet, OpenCV's face detector (MIT licence, 230 KB), downloaded to models/ on
first use. It finds the faces in everyday photos (people from the waist up)
that Florence-2 only labels "person"; Florence-2 still finds the faces in
close-ups, which YuNet misses.
"""
import hashlib
import os
import tempfile
import threading
import urllib.request

import cv2
import numpy as np

from brain.networks import MODELS_DIR

YUNET_FILE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
# Pinned to a commit and checked, so a moved or half-downloaded file isn't used.
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DOWNLOAD_SECONDS = 60  # a stalled download fails instead of hanging startup
FACE_SCORE = 0.8  # YuNet's confidence needed to count as a face


class YuNet:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            str(_yunet_file()), "", (320, 320), score_threshold=FACE_SCORE
        )
        # One detector for every request: its input size is set per picture.
        self.lock = threading.Lock()

    def find_faces(self, image):
        """The boxes [x1, y1, x2, y2] of the faces in a PIL image."""
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        with self.lock:
            self.detector.setInputSize(image.size)
            _, found = self.detector.detect(bgr)
        if found is None:
            return []
        return [[x, y, x + w, y + h] for x, y, w, h in found[:, :4]]


def _yunet_file():
    """
    YuNet's model file. Downloaded on first use to a temp file of this
    process's own, checked, then moved in place: an interrupted or bad
    download is never kept, and processes starting together don't clash.
    """
    if YUNET_FILE.exists():
        return YUNET_FILE
    YUNET_FILE.parent.mkdir(exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=YUNET_FILE.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(YUNET_URL, timeout=DOWNLOAD_SECONDS) as r:
            data = r.read()
            out.write(data)
        if hashlib.sha256(data).hexdigest() != YUNET_SHA256:
            raise RuntimeError(f"{YUNET_URL} didn't download correctly; start again to retry")
        os.replace(temp, YUNET_FILE)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return YUNET_FILE
