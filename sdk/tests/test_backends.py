"""
Tests for tagnify.backends — BaseBackend and OllamaBackend.

OllamaBackend makes real HTTP calls, so we mock httpx.Client
to test it without needing Ollama running. This is standard
practice for any code with external dependencies.

Key concept — mocking:
    patch("tagnify.backends.ollama.httpx.Client") replaces the
    httpx.Client name inside ollama.py with a MagicMock we control.
    We pre-program what the mock returns, then verify our code
    handles those returns correctly.
"""

import pytest
import httpx
from unittest.mock import MagicMock, patch

from tagnify.backends.base import BaseBackend
from tagnify.backends.ollama import OllamaBackend
from tagnify.exceptions import BackendError


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def make_mock_http_client(response_content: str = '{"label": "positive", "confidence": 0.9}'):
    """Build a mock httpx.Client context manager.

    Simulates a successful Ollama response with the given content.
    Handles the context manager protocol (__enter__/__exit__).
    """
    # The response object returned by client.post(...)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": response_content},
        "done": True,
    }
    mock_response.raise_for_status.return_value = None  # no exception = success

    # The client object returned by `with httpx.Client() as client:`
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mock_client.post.return_value = mock_response

    return mock_client


# ═══════════════════════════════════════════════════════════════
# BaseBackend — interface contract
# ═══════════════════════════════════════════════════════════════

class TestBaseBackend:

    def test_cannot_instantiate_directly(self):
        """ABC: BaseBackend cannot be instantiated without implementing complete()."""
        with pytest.raises(TypeError, match="abstract"):
            BaseBackend()

    def test_subclass_without_complete_cannot_instantiate(self):
        """A subclass that doesn't implement complete() is still abstract."""
        class IncompleteBackend(BaseBackend):
            pass  # forgot to implement complete()

        with pytest.raises(TypeError):
            IncompleteBackend()

    def test_subclass_with_complete_can_instantiate(self):
        """A subclass that implements complete() is valid."""
        class MinimalBackend(BaseBackend):
            def complete(self, prompt: str) -> str:
                return "response"

        backend = MinimalBackend()
        assert backend.complete("test") == "response"


# ═══════════════════════════════════════════════════════════════
# OllamaBackend — initialisation
# ═══════════════════════════════════════════════════════════════

class TestOllamaBackendInit:

    def test_default_host(self):
        backend = OllamaBackend(model="qwen2.5:7b")
        assert backend.host == "http://localhost:11434"

    def test_default_timeout(self):
        backend = OllamaBackend(model="qwen2.5:7b")
        assert backend.timeout == 120.0

    def test_default_temperature(self):
        backend = OllamaBackend(model="qwen2.5:7b")
        assert backend.temperature == 0.1

    def test_custom_host(self):
        backend = OllamaBackend(model="qwen2.5:7b", host="http://192.168.1.10:11434")
        assert backend.host == "http://192.168.1.10:11434"

    def test_trailing_slash_stripped(self):
        """Trailing slash on host would cause double-slash in URLs."""
        backend = OllamaBackend(model="qwen2.5:7b", host="http://localhost:11434/")
        assert backend.host == "http://localhost:11434"

    def test_custom_temperature(self):
        backend = OllamaBackend(model="qwen2.5:7b", temperature=0.0)
        assert backend.temperature == 0.0

    def test_is_subclass_of_base_backend(self):
        """OllamaBackend must satisfy the BaseBackend contract."""
        backend = OllamaBackend(model="qwen2.5:7b")
        assert isinstance(backend, BaseBackend)


# ═══════════════════════════════════════════════════════════════
# OllamaBackend — successful complete()
# ═══════════════════════════════════════════════════════════════

class TestOllamaBackendSuccess:

    def test_returns_response_content(self):
        """complete() returns the 'content' field from Ollama's response."""
        expected = '{"label": "positive", "confidence": 0.91}'
        mock_client = make_mock_http_client(response_content=expected)

        with patch("tagnify.backends.ollama.httpx.Client", return_value=mock_client):
            backend = OllamaBackend(model="qwen2.5:7b")
            result = backend.complete("test prompt")

        assert result == expected

    def test_posts_to_correct_url(self):
        """complete() calls the /api/chat endpoint on the configured host."""
        mock_client = make_mock_http_client()

        with patch("tagnify.backends.ollama.httpx.Client", return_value=mock_client):
            backend = OllamaBackend(model="qwen2.5:7b")
            backend.complete("test prompt")

        call_args = mock_client.post.call_args
        actual_url = call_args[0][0]
        assert actual_url == "http://localhost:11434/api/chat"

    def test_sends_correct_model_in_payload(self):
        """The model name in the payload must match what was configured."""
        mock_client = make_mock_http_client()

        with patch("tagnify.backends.ollama.httpx.Client", return_value=mock_client):
            backend = OllamaBackend(model="deepseek-r1:8b")
            backend.complete("test prompt")

        payload = mock_client.post.call_args[1]["json"]
        assert payload["model"] == "deepseek-r1:8b"

    def test_sends_prompt_as_user_message(self):
        """Prompt goes into a user-role message in the messages array."""
        mock_client = make_mock_http_client()

        with patch("tagnify.backends.ollama.httpx.Client", return_value=mock_client):
            backend = OllamaBackend(model="qwen2.5:7b")
            backend.complete("my specific prompt")

        payload = mock_client.post.call_args[1]["json"]
        messages = payload["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "my specific prompt"

    def test_stream_is_disabled(self):
        """Streaming must be False — we want a single complete response."""
        mock_client = make_mock_http_client()

        with patch("tagnify.backends.ollama.httpx.Client", return_value=mock_client):
            backend = OllamaBackend(model="qwen2.5:7b")
            backend.complete("prompt")

        payload = mock_client.post.call_args[1]["json"]
        assert payload["stream"] is False

    def test_temperature_sent_in_options(self):
        """Temperature is passed in the options dict."""
        mock_client = make_mock_http_client()

        with patch("tagnify.backends.ollama.httpx.Client", return_value=mock_client):
            backend = OllamaBackend(model="qwen2.5:7b", temperature=0.0)
            backend.complete("prompt")

        payload = mock_client.post.call_args[1]["json"]
        assert payload["options"]["temperature"] == 0.0


# ═══════════════════════════════════════════════════════════════
# OllamaBackend — error handling
# ═══════════════════════════════════════════════════════════════

class TestOllamaBackendErrors:

    def test_connect_error_raises_backend_error(self):
        """ConnectError (Ollama not running) → BackendError with helpful message."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        with patch("tagnify.backends.ollama.httpx.Client", return_value=mock_client):
            backend = OllamaBackend(model="qwen2.5:7b")
            with pytest.raises(BackendError, match="ollama serve"):
                backend.complete("prompt")

    def test_timeout_raises_backend_error(self):
        """TimeoutException → BackendError with timeout-specific message."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post.side_effect = httpx.TimeoutException("Timed out")

        with patch("tagnify.backends.ollama.httpx.Client", return_value=mock_client):
            backend = OllamaBackend(model="qwen2.5:7b")
            with pytest.raises(BackendError, match="timeout"):
                backend.complete("prompt")

    def test_http_error_raises_backend_error(self):
        """HTTPStatusError (4xx/5xx) → BackendError with status code."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "model not found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found",
            request=MagicMock(),
            response=mock_response,
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post.return_value = mock_response

        with patch("tagnify.backends.ollama.httpx.Client", return_value=mock_client):
            backend = OllamaBackend(model="qwen2.5:7b")
            with pytest.raises(BackendError, match="404"):
                backend.complete("prompt")

    def test_backend_error_is_subclass_of_tagnify_error(self):
        """BackendError can be caught with the base TagnifyError."""
        from tagnify.exceptions import TagnifyError
        assert issubclass(BackendError, TagnifyError)