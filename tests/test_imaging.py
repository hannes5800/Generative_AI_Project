from types import SimpleNamespace

from src import imaging


class DummyImages:
    def __init__(self, expected_prompt: str, url: str = "https://example.com/image.png"):
        self.expected_prompt = expected_prompt
        self.url = url
        self.calls = []

    def generate(self, model: str, prompt: str, size: str, n: int):
        self.calls.append({"model": model, "prompt": prompt, "size": size, "n": n})
        assert prompt == self.expected_prompt
        return SimpleNamespace(data=[SimpleNamespace(url=self.url)])


class DummyClient:
    def __init__(self, images: DummyImages):
        self.images = images


def test_generate_tokenomics_image_success(monkeypatch):
    prompt = "tokenomics flow with arrows"
    dummy_images = DummyImages(prompt)

    monkeypatch.setattr(imaging, "OpenAI", lambda: DummyClient(dummy_images))

    url = imaging.generate_tokenomics_image(prompt)

    assert url == dummy_images.url
    assert dummy_images.calls == [
        {
            "model": "dall-e-3",
            "prompt": prompt,
            "size": "1024x1024",
            "n": 1,
        }
    ]
