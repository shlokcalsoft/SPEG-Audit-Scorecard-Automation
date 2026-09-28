from src import provider_router


def test_empty_primary_response_uses_fallback(monkeypatch):
    class Message:
        content = ""

    class Choice:
        message = Message()

    class Completions:
        def create(self, **kwargs):
            return type("Response", (), {"choices": [Choice()]})()

    client = type("Client", (), {
        "chat": type("Chat", (), {"completions": Completions()})(),
    })()

    monkeypatch.setattr(
        provider_router,
        "request_fallback",
        lambda messages, max_tokens: ("{\"ok\": true}", "openrouter/test"),
    )

    content, provider = provider_router.request_with_fallback(
        client, "groq-model", [], 100
    )

    assert content == '{"ok": true}'
    assert provider == "openrouter/test"