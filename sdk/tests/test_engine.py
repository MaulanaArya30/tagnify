"""
Tests for tagnify.engine — LabelEngine.

Uses a MockBackend that returns pre-programmed responses instead
of making HTTP calls. This lets us precisely control what the
"model" returns on each attempt, including simulating failures.

Key patterns:
    - Success on attempt 1: single valid response
    - Success on attempt 2: one bad response then one valid response
    - Exhaustion: all responses are bad (or too few responses)
    - BackendError: exception in the response list propagates up
"""

import pytest
from tagnify.engine import LabelEngine
from tagnify.backends.base import BaseBackend
from tagnify.schema import Schema, Example, LabelResult
from tagnify.exceptions import BackendError


# ═══════════════════════════════════════════════════════════════
# MockBackend
# ═══════════════════════════════════════════════════════════════

class MockBackend(BaseBackend):
    """Test backend with pre-programmed responses.

    Accepts a list of responses. Each call to complete() pops the
    first response and returns it. If a response is an Exception
    instance, it raises instead of returning.

    Also records every prompt received, so tests can assert on
    prompt content (e.g. that retry reminders are included).
    """

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.prompts_received: list[str] = []

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts_received.append(prompt)
        if not self.responses:
            raise BackendError("MockBackend: no more responses programmed")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

VALID_RESPONSE = '{"label": "positive", "confidence": 0.9}'
NEGATIVE_RESPONSE = '{"label": "negative", "confidence": 0.85}'
LOW_CONFIDENCE_RESPONSE = '{"label": "neutral", "confidence": 0.3}'
BAD_RESPONSE = "I think this is probably positive? Not sure."
INVALID_LABEL_RESPONSE = '{"label": "maybe", "confidence": 0.9}'
REASONING_RESPONSE = (
    '{"label": "positive", "confidence": 0.9, '
    '"reasoning": "Clear positive language."}'
)


def make_schema(**kwargs) -> Schema:
    defaults = {
        "labels": ["positive", "negative", "neutral"],
        "examples": [Example(text="Great!", label="positive")],
        "confidence_threshold": 0.5,
    }
    return Schema(**{**defaults, **kwargs})


def make_engine(responses: list, max_retries: int = 3) -> tuple[LabelEngine, MockBackend]:
    """Build an engine with a MockBackend. Returns both for inspection."""
    backend = MockBackend(responses)
    engine = LabelEngine(backend=backend, max_retries=max_retries)
    return engine, backend


# ═══════════════════════════════════════════════════════════════
# Successful labeling
# ═══════════════════════════════════════════════════════════════

class TestEngineSuccess:

    def test_success_on_first_attempt(self):
        """Happy path — valid response on attempt 1."""
        engine, backend = make_engine([VALID_RESPONSE])
        result = engine.run("Great product!", make_schema())

        assert result.success is True
        assert result.label == "positive"
        assert result.confidence == 0.9
        assert result.attempts == 1
        assert backend.call_count == 1

    def test_success_returns_label_result_type(self):
        engine, _ = make_engine([VALID_RESPONSE])
        result = engine.run("test", make_schema())
        assert isinstance(result, LabelResult)

    def test_success_after_one_retry(self):
        """First attempt returns bad output; second succeeds."""
        engine, backend = make_engine([BAD_RESPONSE, VALID_RESPONSE])
        result = engine.run("Great product!", make_schema())

        assert result.success is True
        assert result.label == "positive"
        assert result.attempts == 2
        assert backend.call_count == 2

    def test_success_on_third_attempt(self):
        """Two bad attempts, then success."""
        engine, backend = make_engine([
            BAD_RESPONSE,
            INVALID_LABEL_RESPONSE,
            VALID_RESPONSE,
        ])
        result = engine.run("Great product!", make_schema())

        assert result.success is True
        assert result.attempts == 3
        assert backend.call_count == 3

    def test_negative_label_returned_correctly(self):
        engine, _ = make_engine([NEGATIVE_RESPONSE])
        result = engine.run("Terrible experience.", make_schema())
        assert result.label == "negative"
        assert result.confidence == 0.85

    def test_reasoning_in_result_when_enabled(self):
        engine, _ = make_engine([REASONING_RESPONSE])
        result = engine.run("Great!", make_schema(), reasoning=True)
        assert result.reasoning == "Clear positive language."

    def test_reasoning_none_when_disabled(self):
        engine, _ = make_engine([VALID_RESPONSE])
        result = engine.run("Great!", make_schema(), reasoning=False)
        assert result.reasoning is None

    def test_success_has_no_error(self):
        engine, _ = make_engine([VALID_RESPONSE])
        result = engine.run("test", make_schema())
        assert result.error is None


# ═══════════════════════════════════════════════════════════════
# Flagging logic
# ═══════════════════════════════════════════════════════════════

class TestEngineFlagging:

    def test_not_flagged_when_confidence_above_threshold(self):
        """0.9 > 0.5 threshold — should not be flagged."""
        schema = make_schema(confidence_threshold=0.5)
        engine, _ = make_engine([VALID_RESPONSE])  # confidence=0.9
        result = engine.run("test", schema)
        assert result.flagged is False

    def test_flagged_when_confidence_below_threshold(self):
        """0.3 < 0.5 threshold — should be flagged."""
        schema = make_schema(confidence_threshold=0.5)
        engine, _ = make_engine([LOW_CONFIDENCE_RESPONSE])  # confidence=0.3
        result = engine.run("test", schema)
        assert result.flagged is True
        assert result.success is True  # flagged but not failed
        assert result.label == "neutral"  # label is still present

    def test_not_flagged_when_confidence_equals_threshold(self):
        """Boundary: confidence exactly AT threshold is not flagged."""
        at_threshold = '{"label": "positive", "confidence": 0.5}'
        schema = make_schema(confidence_threshold=0.5)
        engine, _ = make_engine([at_threshold])
        result = engine.run("test", schema)
        assert result.flagged is False  # 0.5 < 0.5 is False

    def test_high_threshold_flags_medium_confidence(self):
        """Threshold=0.95 means 0.9 confidence gets flagged."""
        schema = make_schema(confidence_threshold=0.95)
        engine, _ = make_engine([VALID_RESPONSE])  # confidence=0.9
        result = engine.run("test", schema)
        assert result.flagged is True

    def test_zero_threshold_never_flags(self):
        """Threshold=0.0 means nothing gets flagged."""
        schema = make_schema(confidence_threshold=0.0)
        engine, _ = make_engine([LOW_CONFIDENCE_RESPONSE])  # confidence=0.3
        result = engine.run("test", schema)
        assert result.flagged is False


# ═══════════════════════════════════════════════════════════════
# Retry behaviour
# ═══════════════════════════════════════════════════════════════

class TestEngineRetry:

    def test_retry_reminder_in_prompt_on_attempt_2(self):
        """Prompt on attempt 2 should contain the IMPORTANT reminder."""
        engine, backend = make_engine([BAD_RESPONSE, VALID_RESPONSE])
        engine.run("test", make_schema())

        assert len(backend.prompts_received) == 2
        assert "IMPORTANT" in backend.prompts_received[1]

    def test_critical_reminder_in_prompt_on_attempt_3(self):
        """Prompt on attempt 3 should contain the CRITICAL reminder."""
        engine, backend = make_engine([
            BAD_RESPONSE, BAD_RESPONSE, VALID_RESPONSE
        ])
        engine.run("test", make_schema())

        assert len(backend.prompts_received) == 3
        assert "CRITICAL" in backend.prompts_received[2]

    def test_no_reminder_on_first_attempt_prompt(self):
        """No retry reminder should appear in a first-attempt prompt."""
        engine, backend = make_engine([VALID_RESPONSE])
        engine.run("test", make_schema())

        assert "IMPORTANT" not in backend.prompts_received[0]
        assert "CRITICAL" not in backend.prompts_received[0]

    def test_backend_called_once_on_success(self):
        engine, backend = make_engine([VALID_RESPONSE])
        engine.run("test", make_schema())
        assert backend.call_count == 1

    def test_backend_called_twice_on_one_retry(self):
        engine, backend = make_engine([BAD_RESPONSE, VALID_RESPONSE])
        engine.run("test", make_schema())
        assert backend.call_count == 2

    def test_max_retries_respected(self):
        """Engine never calls backend more than max_retries times."""
        engine, backend = make_engine(
            [BAD_RESPONSE] * 10,  # more responses than retries
            max_retries=2,
        )
        engine.run("test", make_schema())
        assert backend.call_count == 2


# ═══════════════════════════════════════════════════════════════
# Failure cases — all retries exhausted
# ═══════════════════════════════════════════════════════════════

class TestEngineExhaustion:

    def test_returns_failure_result_on_exhaustion(self):
        """When all retries fail, return LabelResult(success=False)."""
        engine, _ = make_engine([BAD_RESPONSE, BAD_RESPONSE, BAD_RESPONSE])
        result = engine.run("test", make_schema())
        assert result.success is False

    def test_failure_result_has_null_label(self):
        engine, _ = make_engine([BAD_RESPONSE] * 3)
        result = engine.run("test", make_schema())
        assert result.label is None

    def test_failure_result_is_always_flagged(self):
        """Failed results are always flagged, regardless of threshold."""
        engine, _ = make_engine([BAD_RESPONSE] * 3)
        result = engine.run("test", make_schema(confidence_threshold=0.0))
        assert result.flagged is True

    def test_failure_result_has_error_message(self):
        engine, _ = make_engine([BAD_RESPONSE] * 3)
        result = engine.run("test", make_schema())
        assert result.error is not None
        assert len(result.error) > 0

    def test_failure_result_attempts_equals_max_retries(self):
        engine, _ = make_engine([BAD_RESPONSE] * 3, max_retries=3)
        result = engine.run("test", make_schema())
        assert result.attempts == 3

    def test_failure_with_invalid_label_includes_last_error(self):
        """Error message should describe the final failure."""
        engine, _ = make_engine([INVALID_LABEL_RESPONSE] * 3)
        result = engine.run("test", make_schema())
        assert result.error is not None
        # The last error came from Validator — should mention label issue
        assert "Last error:" in result.error

    def test_does_not_raise_on_exhaustion(self):
        """The engine must return, not raise, when retries are exhausted."""
        engine, _ = make_engine([BAD_RESPONSE] * 3)
        # If this raises, the test fails — exactly what we want to prevent
        result = engine.run("test", make_schema())
        assert result.success is False


# ═══════════════════════════════════════════════════════════════
# BackendError handling
# ═══════════════════════════════════════════════════════════════

class TestEngineBackendError:

    def test_backend_error_propagates(self):
        """BackendError is not caught — it propagates to the caller."""
        engine, _ = make_engine([BackendError("Ollama is not running")])
        with pytest.raises(BackendError, match="Ollama is not running"):
            engine.run("test", make_schema())

    def test_backend_error_not_retried(self):
        """After BackendError, backend should not be called again."""
        engine, backend = make_engine([
            BackendError("Connection refused"),
            VALID_RESPONSE,  # this should never be reached
        ])
        with pytest.raises(BackendError):
            engine.run("test", make_schema())
        assert backend.call_count == 1  # only called once


# ═══════════════════════════════════════════════════════════════
# Engine initialisation
# ═══════════════════════════════════════════════════════════════

class TestEngineInit:

    def test_max_retries_zero_raises(self):
        """max_retries must be at least 1."""
        backend = MockBackend([])
        with pytest.raises(ValueError, match="at least 1"):
            LabelEngine(backend=backend, max_retries=0)

    def test_max_retries_negative_raises(self):
        backend = MockBackend([])
        with pytest.raises(ValueError, match="at least 1"):
            LabelEngine(backend=backend, max_retries=-1)

    def test_custom_max_retries(self):
        engine, backend = make_engine([BAD_RESPONSE, BAD_RESPONSE], max_retries=2)
        result = engine.run("test", make_schema())
        assert backend.call_count == 2