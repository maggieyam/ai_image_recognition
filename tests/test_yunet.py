import pytest

from brain.networks import yunet


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
    monkeypatch.setattr(yunet, "YUNET_FILE", tmp_path / "yunet.onnx")
    monkeypatch.setattr(yunet.urllib.request, "urlopen", lambda url, timeout: FakeResponse(data))
    with pytest.raises((RuntimeError, OSError)):
        yunet._yunet_file()
    assert list(tmp_path.iterdir()) == []  # neither the model nor a leftover temp file
