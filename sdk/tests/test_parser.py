"""
Tests for tagnify.parser — OutputParser.

OutputParser is pure string manipulation — no HTTP, no backends,
no external dependencies. Tests simply pass strings in and assert
on the resulting dict or exception.

Test strings are written to match realistic LLM outputs, because
the whole point of this class is to handle what models actually produce.
"""

import pytest
from tagnify.parser import OutputParser
from tagnify.exceptions import OutputParseError


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def parse(text: str) -> dict:
    """Convenience wrapper — removes OutputParser() boilerplate from every test."""
    return OutputParser().parse(text)


# ═══════════════════════════════════════════════════════════════
# Strategy 1: direct JSON parse
# ═══════════════════════════════════════════════════════════════

class TestDirectParse:

    def test_clean_json_object(self):
        """Happy path — model followed instructions perfectly."""
        result = parse('{"label": "positive", "confidence": 0.9}')
        assert result == {"label": "positive", "confidence": 0.9}

    def test_clean_json_with_reasoning(self):
        result = parse(
            '{"label": "positive", "confidence": 0.9, "reasoning": "Clearly positive."}'
        )
        assert result["label"] == "positive"
        assert result["reasoning"] == "Clearly positive."

    def test_leading_trailing_whitespace_stripped(self):
        """Whitespace around the JSON is cleaned before parsing."""
        result = parse('   {"label": "negative", "confidence": 0.85}   ')
        assert result["label"] == "negative"

    def test_extra_fields_preserved(self):
        """Unknown extra keys pass through — Validator decides what to keep."""
        result = parse('{"label": "positive", "confidence": 0.9, "extra": "ignored"}')
        assert result["label"] == "positive"
        assert "extra" in result


# ═══════════════════════════════════════════════════════════════
# Strategy 2: strip markdown fences
# ═══════════════════════════════════════════════════════════════

class TestMarkdownFenceStripping:

    def test_json_code_fence(self):
        """```json...``` is the most common markdown wrapper."""
        result = parse('```json\n{"label": "positive", "confidence": 0.9}\n```')
        assert result["label"] == "positive"

    def test_plain_code_fence(self):
        """``` without language tag is also common."""
        result = parse('```\n{"label": "negative", "confidence": 0.8}\n```')
        assert result["label"] == "negative"

    def test_uppercase_json_fence(self):
        """Some models use ```JSON (uppercase)."""
        result = parse('```JSON\n{"label": "neutral", "confidence": 0.6}\n```')
        assert result["label"] == "neutral"

    def test_fence_with_no_newline(self):
        """Fence directly adjacent to JSON content."""
        result = parse('```json{"label": "positive", "confidence": 0.9}```')
        assert result["label"] == "positive"


# ═══════════════════════════════════════════════════════════════
# Strategy 3: extract JSON block from surrounding text
# ═══════════════════════════════════════════════════════════════

class TestJsonBlockExtraction:

    def test_text_before_json(self):
        """Common when model adds a preamble before the JSON."""
        result = parse(
            'Based on the text, here is my classification:\n'
            '{"label": "neutral", "confidence": 0.6}'
        )
        assert result["label"] == "neutral"

    def test_text_after_json(self):
        """Common when model adds explanation after the JSON."""
        result = parse(
            '{"label": "positive", "confidence": 0.95}\n'
            'The text contains clear positive sentiment.'
        )
        assert result["label"] == "positive"

    def test_text_on_both_sides(self):
        """Model wraps the JSON with explanation on both sides."""
        result = parse(
            'My analysis:\n'
            '{"label": "negative", "confidence": 0.88}\n'
            'I am confident in this classification.'
        )
        assert result["label"] == "negative"

    def test_json_inside_list_extracts_first_object(self):
        """Model wraps the object in a JSON array — we extract the inner dict."""
        result = parse('[{"label": "positive", "confidence": 0.9}]')
        assert result["label"] == "positive"

    def test_nested_object_returned_whole(self):
        """Nested JSON objects are preserved correctly."""
        result = parse(
            '{"label": "positive", "confidence": 0.9, "meta": {"source": "test"}}'
        )
        assert result["label"] == "positive"
        assert result["meta"] == {"source": "test"}

    def test_only_first_complete_object_extracted(self):
        """When multiple JSON objects exist, the first complete one is returned."""
        result = parse(
            '{"label": "positive", "confidence": 0.9} {"label": "negative", "confidence": 0.1}'
        )
        assert result["label"] == "positive"


# ═══════════════════════════════════════════════════════════════
# Strategy 4: fix common syntax issues
# ═══════════════════════════════════════════════════════════════

class TestCommonSyntaxFixes:

    def test_single_quoted_keys_and_values(self):
        """Some models use Python dict syntax with single quotes."""
        result = parse("{'label': 'positive', 'confidence': 0.9}")
        assert result["label"] == "positive"
        assert result["confidence"] == 0.9

    def test_trailing_comma(self):
        """Some models add a trailing comma after the last field."""
        result = parse('{"label": "positive", "confidence": 0.9,}')
        assert result["label"] == "positive"

    def test_single_quotes_and_trailing_comma_combined(self):
        """Both issues at once."""
        result = parse("{'label': 'negative', 'confidence': 0.8,}")
        assert result["label"] == "negative"


# ═══════════════════════════════════════════════════════════════
# Failure cases — all strategies exhausted
# ═══════════════════════════════════════════════════════════════

class TestOutputParserFailures:

    def test_empty_string_raises(self):
        with pytest.raises(OutputParseError):
            parse("")

    def test_whitespace_only_raises(self):
        with pytest.raises(OutputParseError):
            parse("   \n\t  ")

    def test_plain_text_raises(self):
        """Model ignored format entirely and returned natural language."""
        with pytest.raises(OutputParseError):
            parse("I think this text is positive with high confidence.")

    def test_partial_json_raises(self):
        """Truncated or broken JSON that can't be recovered."""
        with pytest.raises(OutputParseError):
            parse('{"label": "positive", "confide')

    def test_raises_output_parse_error_type(self):
        """Exception type must be OutputParseError specifically."""
        with pytest.raises(OutputParseError):
            parse("not json at all")

    def test_output_parse_error_is_tagnify_error(self):
        """OutputParseError can be caught with the base TagnifyError."""
        from tagnify.exceptions import TagnifyError
        assert issubclass(OutputParseError, TagnifyError)