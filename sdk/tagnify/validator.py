#checks if the tag is valid

from tagnify.schema import Schema
from tagnify.exceptions import ValidationError

class Validator:


    #validation functions
    def _validate_label(self, parsed: dict, schema: Schema) -> dict:
        label = parsed.get("label")
        if label is None:
            raise ValidationError(
                f'LLM output is missing the "label" field. '
                f'Got keys: {list(parsed.keys())}. '
                f'Expected format: {{"label": "<label>", "confidence": <0.0-1.0>}}'
            )
        if not isinstance(label, str):
            raise ValidationError(
                f'"label" must be a string, got '
                f'{type(label).__name__}: {label!r}'
            )
        
        label = label.strip()
        if label not in schema.labels:
            raise ValidationError(
                f'Label "{label}" is not valid. '
                f'Must be exactly one of: {schema.labels}. '
                f'Labels are case-sensitive and must match exactly.'
            )

        return label
    
    def _validate_confidence(self, parsed: dict) -> float:
        confidence_raw = parsed.get("confidence")
        if confidence_raw is None:
            raise ValidationError(
                f'LLM output is missing the "confidence" field. '
                f'Got keys: {list(parsed.keys())}. '
                f'Expected format: {{"label": "<label>", "confidence": <0.0-1.0>}}'
            )

        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            raise ValidationError(
                f'"confidence" must be a number between 0.0 and 1.0, '
                f'got: {confidence_raw!r} (type: {type(confidence_raw).__name__}). '
                f'Example: {{"label": "positive", "confidence": 0.9}}'
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValidationError(
                f'"confidence" must be between 0.0 and 1.0, got {confidence}. '
                f'Confidence is a probability and must stay within this range.'
            )

        return confidence
    