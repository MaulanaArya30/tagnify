"""
Tests for tagnify.validator — Validator.

Validator receives a parsed dict from OutputParser and checks it
against the Schema. Tests pass pre-built dicts directly, since
OutputParser is tested separately.

Covers:
    - Valid inputs across all field types
    - Each individual failure mode (missing, wrong type, out of range)
    - Type coercion for confidence (string numbers, ints)
    - Whitespace stripping on labels
    - Reasoning field handling (optional, coercion)
"""

import pytest
from tagnify.validator import Validator
from tagnify.schema import Schema, Example
from tagnify.exceptions import ValidationError, TagnifyError


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def make_schema(**kwargs) -> Schema:
    """Minimal valid schema with overridable fields."""
    defaults = {
        "labels": ["positive", "negative", "neutral"],
        "examples": [Example(text="Great!", label="positive")],
    }
    return Schema(**{**defaults, **kwargs})


def validate(parsed: dict, schema: Schema = None) -> dict:
    """Convenience wrapper."""
    if schema is None:
        schema = make_schema()
    return Validator().validate(parsed, schema)


# ═══════════════════════════════════════════════════════════════
# Valid inputs
# ═══════════════════════════════════════════════════════════════

class TestValidatorSuccess:

    def test_basic_valid_input(self):
        """Happy path — model returned exactly what was asked."""
        result = validate({"label": "positive", "confidence": 0.9})
        assert result["label"] == "positive"
        assert result["confidence"] == 0.9
        assert result["reasoning"] is None

    def test_returns_exactly_three_keys(self):
        """Return dict always has label, confidence, reasoning — no more, no less."""
        result = validate({"label": "negative", "confidence": 0.75})
        assert set(result.keys()) == {"label", "confidence", "reasoning"}

    def test_all_valid_labels_accepted(self):
        """Each label in the schema can be returned."""
        schema = make_schema()
        for label in schema.labels:
            result = validate({"label": label, "confidence": 0.8}, schema)
            assert result["label"] == label

    def test_reasoning_included_when_present(self):
        result = validate({
            "label": "positive",
            "confidence": 0.9,
            "reasoning": "Strong positive language.",
        })
        assert result["reasoning"] == "Strong positive language."

    def test_reasoning_none_when_absent(self):
        """Reasoning is optional — absence is not a failure."""
        result = validate({"label": "negative", "confidence": 0.8})
        assert result["reasoning"] is None

    def test_confidence_as_int_coerced_to_float(self):
        """Confidence of 1 (integer) should become 1.0 (float)."""
        result = validate({"label": "positive", "confidence": 1})
        assert result["confidence"] == 1.0
        assert isinstance(result["confidence"], float)

    def test_confidence_as_string_number_coerced(self):
        """Some models return confidence as a string: "0.9" → 0.9."""
        result = validate({"label": "positive", "confidence": "0.9"})
        assert result["confidence"] == 0.9

    def test_confidence_boundary_zero(self):
        """0.0 is a valid confidence (completely uncertain)."""
        result = validate({"label": "neutral", "confidence": 0.0})
        assert result["confidence"] == 0.0

    def test_confidence_boundary_one(self):
        """1.0 is a valid confidence (perfectly certain)."""
        result = validate({"label": "positive", "confidence": 1.0})
        assert result["confidence"] == 1.0

    def test_label_whitespace_stripped(self):
        """Leading/trailing whitespace on label is stripped before checking."""
        result = validate({"label": " positive ", "confidence": 0.9})
        assert result["label"] == "positive"

    def test_extra_fields_in_parsed_ignored(self):
        """Extra keys in the parsed dict don't cause failures."""
        result = validate({
            "label": "negative",
            "confidence": 0.85,
            "extra_field": "ignored",
            "another": 123,
        })
        assert result["label"] == "negative"
        assert "extra_field" not in result


# ═══════════════════════════════════════════════════════════════
# Label failures
# ═══════════════════════════════════════════════════════════════

class TestValidatorLabelFailures:

    def test_missing_label_raises(self):
        """Label field absent entirely."""
        with pytest.raises(ValidationError, match='"label"'):
            validate({"confidence": 0.9})

    def test_null_label_raises(self):
        """Label field present but null (None)."""
        with pytest.raises(ValidationError, match='"label"'):
            validate({"label": None, "confidence": 0.9})

    def test_label_not_in_schema_raises(self):
        """Label value doesn't match any schema label."""
        with pytest.raises(ValidationError, match="not valid"):
            validate({"label": "maybe", "confidence": 0.9})

    def test_label_wrong_case_raises(self):
        """Labels are case-sensitive — 'Positive' ≠ 'positive'."""
        with pytest.raises(ValidationError, match="not valid"):
            validate({"label": "Positive", "confidence": 0.9})

    def test_label_wrong_type_int_raises(self):
        """Label must be a string, not an integer."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate({"label": 1, "confidence": 0.9})

    def test_label_wrong_type_bool_raises(self):
        """Label must be a string, not a boolean."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate({"label": True, "confidence": 0.9})

    def test_label_empty_string_raises(self):
        """Empty string is not in the labels list."""
        with pytest.raises(ValidationError, match="not valid"):
            validate({"label": "", "confidence": 0.9})

    def test_label_whitespace_only_raises(self):
        """Whitespace-only label strips to empty string, not in labels."""
        with pytest.raises(ValidationError, match="not valid"):
            validate({"label": "   ", "confidence": 0.9})


# ═══════════════════════════════════════════════════════════════
# Confidence failures
# ═══════════════════════════════════════════════════════════════

class TestValidatorConfidenceFailures:

    def test_missing_confidence_raises(self):
        """Confidence field absent entirely."""
        with pytest.raises(ValidationError, match='"confidence"'):
            validate({"label": "positive"})

    def test_null_confidence_raises(self):
        """Confidence field present but null (None)."""
        with pytest.raises(ValidationError, match='"confidence"'):
            validate({"label": "positive", "confidence": None})

    def test_confidence_non_numeric_string_raises(self):
        """String like 'high' cannot be converted to float."""
        with pytest.raises(ValidationError, match="must be a number"):
            validate({"label": "positive", "confidence": "high"})

    def test_confidence_too_high_raises(self):
        """Confidence above 1.0 is not a valid probability."""
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            validate({"label": "positive", "confidence": 1.5})

    def test_confidence_too_low_raises(self):
        """Confidence below 0.0 is not a valid probability."""
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            validate({"label": "positive", "confidence": -0.1})

    def test_confidence_list_raises(self):
        """Confidence as a list cannot be converted to float."""
        with pytest.raises(ValidationError, match="must be a number"):
            validate({"label": "positive", "confidence": [0.9]})


# ═══════════════════════════════════════════════════════════════
# Exception hierarchy
# ═══════════════════════════════════════════════════════════════

class TestValidatorExceptionHierarchy:

    def test_raises_validation_error_type(self):
        """Must raise ValidationError specifically, not a generic exception."""
        with pytest.raises(ValidationError):
            validate({"label": "not_valid_label", "confidence": 0.9})

    def test_validation_error_is_tagnify_error(self):
        """ValidationError can be caught with the base TagnifyError."""
        assert issubclass(ValidationError, TagnifyError)

    def test_validation_error_message_includes_useful_info(self):
        """Error message should help debug what went wrong."""
        try:
            validate({"label": "wrong", "confidence": 0.9})
        except ValidationError as e:
            assert "wrong" in str(e)
            assert "not valid" in str(e)