"""Testes da API (tradutor falso) + teste de integração opcional com o modelo real.

Integração:  RUN_INTEGRATION=1 pytest
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_translator
from app.translator import TranslationResult


class FakeTranslator:
    model_name = "fake-opus-mt-en-de"

    def translate(self, text: str) -> TranslationResult:
        return TranslationResult(text=f"[de] {text}", chunks=1)


@pytest.fixture
def client():
    app.dependency_overrides[get_translator] = lambda: FakeTranslator()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_translate_returns_json(client):
    body = client.post("/translate", json={"text": "Hello"}).json()
    assert body["translated_text"] == "[de] Hello"
    assert (body["source_language"], body["target_language"]) == ("en", "de")
    assert body["model"] == "fake-opus-mt-en-de"


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": "a" * 10_001}])
def test_translate_rejects_invalid_text(client, payload):
    assert client.post("/translate", json=payload).status_code == 422


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="defina RUN_INTEGRATION=1")
def test_translate_with_real_model():
    long_text = "\n\n".join(["The weather is nice today, so we decided to go for a walk in the park. " * 15] * 2)
    with TestClient(app) as real_client:
        short = real_client.post("/translate", json={"text": "Good morning, my friend."}).json()
        long = real_client.post("/translate", json={"text": long_text}).json()

    assert "guten morgen" in short["translated_text"].lower()
    assert long["chunks"] >= 2
    assert long["translated_text"].count("\n\n") == 1
