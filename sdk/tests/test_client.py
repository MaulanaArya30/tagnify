"""
Tests for tagnify.client — Tagnify.

The client is thin — it selects a backend and delegates to the engine.
Tests verify the wiring: correct backend selection, correct delegation,
and correct batch behaviour.

We use MockBackend from test_engine.py patterns rather than real Ollama.
"""

import pytest
from tagnify.client import Tagnify
from tagnify.backends.ollama import OllamaBackend
from tagnify.schema import Schema, Example, LabelResult
from tagnify.engine import LabelEngine
from tagnify.backends.base import BaseBackend


# ═══════════════════════════════════════════════════════════════
# Helpers — reusing the same MockBackend pattern
# ═══════════════════════════════════════════════════════════════

class MockBackend(BaseBackend):
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        if not self.responses:
            raise Exception("MockBackend: out of responses")
        return self.responses.pop(0)


def make_schema() -> Schema:
    return Schema(
        labels=["positive", "negative"],
        examples=[Example(text="Great!", label="positive")],
    )


def make_tagnify_with_mock(responses: list) -> tuple[Tagnify, MockBackend]:
    """Build a Tagnify instance that uses a MockBackend internally."""
    mock = MockBackend(responses)
    client = Tagnify.__new__(Tagnify)  # bypass __init__
    client._backend = mock
    client._engine = LabelEngine(backend=mock, max_retries=3)
    return client, mock


VALID_RESPONSE = '{"label": "positive", "confidence": 0.9}'
BAD_RESPONSE = "not json"


# ═══════════════════════════════════════════════════════════════
# Backend selection
# ═══════════════════════════════════════════════════════════════

class TestClientBackendSelection:

    def test_no_api_key_creates_ollama_backend(self):
        """api_key=None (default) → OllamaBackend."""
        client = Tagnify(model="qwen2.5:7b")
        assert isinstance(client._backend, OllamaBackend)

    def test_ollama_backend_has_correct_model(self):
        client = Tagnify(model="deepseek-r1:8b")
        assert client._backend.model == "deepseek-r1:8b"

    def test_ollama_backend_has_custom_host(self):
        client = Tagnify(model="qwen2.5:7b", ollama_host="http://0.0.0.0:11434")
        assert client._backend.host == "http://0.0.0.0:11434"

    def test_api_key_raises_not_implemented(self):
        """Cloud backend is post-MVP — should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="coming soon"):
            Tagnify(model="qwen2.5:7b", api_key="tgnf-test-key")

    def test_engine_is_created(self):
        client = Tagnify(model="qwen2.5:7b")
        assert isinstance(client._engine, LabelEngine)


# ═══════════════════════════════════════════════════════════════
# label()
# ═══════════════════════════════════════════════════════════════

class TestClientLabel:

    def test_label_returns_label_result(self):
        client, _ = make_tagnify_with_mock([VALID_RESPONSE])
        result = client.label("Great product!", make_schema())
        assert isinstance(result, LabelResult)

    def test_label_returns_correct_label(self):
        client, _ = make_tagnify_with_mock([VALID_RESPONSE])
        result = client.label("Great product!", make_schema())
        assert result.label == "positive"

    def test_label_success_is_true(self):
        client, _ = make_tagnify_with_mock([VALID_RESPONSE])
        result = client.label("Great product!", make_schema())
        assert result.success is True

    def test_label_failure_returns_result_not_raise(self):
        """Even complete failure returns a LabelResult, doesn't raise."""
        client, _ = make_tagnify_with_mock([BAD_RESPONSE] * 3)
        result = client.label("test", make_schema())
        assert result.success is False


# ═══════════════════════════════════════════════════════════════
# label_batch()
# ═══════════════════════════════════════════════════════════════

class TestClientLabelBatch:

    def test_label_batch_returns_list(self):
        client, _ = make_tagnify_with_mock([VALID_RESPONSE, VALID_RESPONSE])
        results = client.label_batch(["text one", "text two"], make_schema())
        assert isinstance(results, list)

    def test_label_batch_length_matches_input(self):
        client, _ = make_tagnify_with_mock([VALID_RESPONSE] * 3)
        results = client.label_batch(["a", "b", "c"], make_schema())
        assert len(results) == 3

    def test_label_batch_all_are_label_results(self):
        client, _ = make_tagnify_with_mock([VALID_RESPONSE] * 2)
        results = client.label_batch(["a", "b"], make_schema())
        assert all(isinstance(r, LabelResult) for r in results)

    def test_label_batch_one_failure_does_not_stop_batch(self):
        """Failure on item 1 should not prevent item 2 from being labeled."""
        client, _ = make_tagnify_with_mock([
            BAD_RESPONSE, BAD_RESPONSE, BAD_RESPONSE,  # item 1 fails
            VALID_RESPONSE,                             # item 2 succeeds
        ])
        results = client.label_batch(["fail", "succeed"], make_schema())
        assert results[0].success is False
        assert results[1].success is True

    def test_empty_batch_returns_empty_list(self):
        client, _ = make_tagnify_with_mock([])
        results = client.label_batch([], make_schema())
        assert results == []


# ═══════════════════════════════════════════════════════════════
# with_backend() — custom backend support
# ═══════════════════════════════════════════════════════════════

class TestClientWithBackend:

    def test_returns_tagnify_instance(self):
        """with_backend() must return an actual Tagnify, not something else."""
        backend = MockBackend([VALID_RESPONSE])
        client = Tagnify.with_backend(backend)
        assert isinstance(client, Tagnify)

    def test_uses_the_exact_backend_provided(self):
        """The client must hold a reference to the same backend instance —
        not a copy, not a rebuilt equivalent."""
        backend = MockBackend([VALID_RESPONSE])
        client = Tagnify.with_backend(backend)
        assert client._backend is backend

    def test_creates_a_labeling_engine(self):
        backend = MockBackend([VALID_RESPONSE])
        client = Tagnify.with_backend(backend)
        assert isinstance(client._engine, LabelEngine)

    def test_label_works_end_to_end_with_custom_backend(self):
        """The whole point: label() must work identically to the
        built-in backends once wired through with_backend()."""
        backend = MockBackend([VALID_RESPONSE])
        client = Tagnify.with_backend(backend)
        result = client.label("Great product!", make_schema())
        assert result.success is True
        assert result.label == "positive"

    def test_label_batch_works_with_custom_backend(self):
        backend = MockBackend([VALID_RESPONSE, VALID_RESPONSE])
        client = Tagnify.with_backend(backend)
        results = client.label_batch(["a", "b"], make_schema())
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_custom_max_retries_is_respected(self):
        """max_retries on with_backend() should reach the engine,
        same as it does via the normal __init__ path."""
        backend = MockBackend([BAD_RESPONSE, BAD_RESPONSE])
        client = Tagnify.with_backend(backend, max_retries=2)
        result = client.label("test", make_schema())
        assert result.attempts == 2
        assert backend.call_count == 2

    def test_default_max_retries_is_three(self):
        backend = MockBackend([BAD_RESPONSE, BAD_RESPONSE, BAD_RESPONSE])
        client = Tagnify.with_backend(backend)
        result = client.label("test", make_schema())
        assert result.attempts == 3

    def test_rejects_non_basebackend_instance(self):
        """Passing something that isn't a BaseBackend should fail loudly
        and immediately — not silently break deep inside the engine later."""
        class NotABackend:
            def complete(self, prompt: str) -> str:
                return VALID_RESPONSE

        with pytest.raises(TypeError, match="BaseBackend instance"):
            Tagnify.with_backend(NotABackend())

    def test_rejects_plain_string(self):
        with pytest.raises(TypeError, match="BaseBackend instance"):
            Tagnify.with_backend("not a backend at all")

    def test_rejects_none(self):
        with pytest.raises(TypeError, match="BaseBackend instance"):
            Tagnify.with_backend(None)

    def test_error_message_includes_actual_type_received(self):
        """Error should name the wrong type, for fast debugging."""
        with pytest.raises(TypeError, match="str"):
            Tagnify.with_backend("oops")