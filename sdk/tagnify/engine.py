#engine, contact point for all operations with added retry logic

from tagnify.backends.base import BaseBackend
from tagnify.schema import Schema, LabelResult
from tagnify.prompt import PromptBuilder
from tagnify.parser import OutputParser
from tagnify.validator import Validator
from tagnify.exceptions import ValidationError, BackendError, OutputParserError



class LabelEngine:
    """Main labeling pipeline"""
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        backend: BaseBackend,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be at least 1, got {max_retries}")
        
        self.backend = backend
        self.max_retries = max_retries
        self._prompt_builder = PromptBuilder()
        self._parser = OutputParser()
        self._validator = Validator()

    
    def run(
        self,
        text: str,
        schema: Schema,
        reasoning: bool = False,
    ) -> LabelResult:
        """Run the labeling pipeline for one item"""
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                #build prompt
                prompt = self._prompt_builder.build(text=text, schema=schema, reasoning=reasoning, attempt=attempt)

                #call backend
                raw_text = self.backend.complete(prompt)
                
                #parse output
                parsed = self._parser.parse(raw_text)
        
                #validate output
                validated = self._validator.validate(parsed, schema)

                #return label result
                confidence = validated['confidence']
                return LabelResult(
                    label=validated['label'],
                    confidence=confidence,
                    reasoning=validated.get('reasoning'),
                    flagged=confidence < schema.confidence_threshold if confidence is not None else False,
                    attempts=attempt,
                    success=True,
                )
            except BackendError:
                raise
            except (OutputParserError, ValidationError) as e:
                last_error = e
        
        return LabelResult(
            label=None,
            confidence=0.0,
            reasoning=None,
            flagged=True,
            attempts=self.max_retries,
            success=False,
             error=(
                f"Labeling failed after {self.max_retries} "
                f"attempt{'s' if self.max_retries > 1 else ''}. "
                f"Last error: {last_error}"
            ),
        )
                