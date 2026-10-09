from types import SimpleNamespace

from app.generation.claude_client import generate_answer
from app.vectorstore.base import RetrievedChunk


class FakeClaude:
    def __init__(self):
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Part one. "),
                SimpleNamespace(type="text", text="Part two."),
            ]
        )


def test_generate_answer_builds_request_and_returns_text():
    fake = FakeClaude()
    chunks = [
        RetrievedChunk("Chunk body about caching.", "d1", "Cache Doc", "Intro", 0.1)
    ]
    answer = generate_answer("How does caching work?", chunks, client=fake, model="m")
    assert answer == "Part one. Part two."
    req = fake.requests[0]
    assert req["model"] == "m"
    content = req["messages"][0]["content"]
    assert "How does caching work?" in content
    assert "Chunk body about caching." in content
    assert "Cache Doc" in content and "Intro" in content
    assert "only" in req["system"]
