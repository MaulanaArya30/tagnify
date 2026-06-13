"""
Turns schema into prompt string

Section 1: role and task description, including schema description if provided.
Section 2: few-shot examples from the schema. If reasoning=True and an example has reasoning text, that reasoning is included as an extra teaching signal.
Section 3: the actual text to label.
Section 4: exact JSON output format and rules. This section is the primary defence against malformed output. Every rule targets a specific common failure mode.
Section 5 (retry only): escalating format reminder. Attempt 2: firm but measured — "IMPORTANT: wrong format." Attempt 3+: maximum specificity — "Start with { end with }."

"""

from tagnify.schema import Schema


class PromptBuilder:
    def build(
        self,
        text: str,
        schema: Schema,
        reasoning: bool = False,
        attempt: int = 1,
    ) -> str:
        """Builds the complete prompt string for the LLM, given the text and the schema.

        Args:
            text: The input text to be tagged.
            schema: The schema defining the tags and their descriptions.
            reasoning: Whether to include reasoning steps in the prompt.
            attempt: The current attempt number (for iterative prompting).
        """
        sections = [
            self._header(schema),
            self._examples(schema, reasoning),
            self._item(text),
            self._format_instruction(schema, reasoning),
        ]
        if attempt > 1:
            sections.append(self._retry_reminder(attempt))
        return "\n\n".join(sections)
    

# -- functions --


    def _header(self, schema: Schema) -> str:
            """Section 1"""
            lines = [
                "You are a precise data labeling assistant.",
                "Your task is to assign exactly one label to the given text.",
            ]
            if schema.description:
                lines.append(f"\nTask context: {schema.description}")
            labels_str = ", ".join(schema.labels)
            lines.append(f"\nValid labels: {labels_str}")
            return "\n".join(lines)

    def _examples(self, schema: Schema, reasoning: bool) -> str:
        """Section 2"""
        lines = ["Examples:"]
        for ex in schema.examples:
            lines.append(f'Text: "{ex.text}"')
            lines.append(f"Label: {ex.label}")
            if reasoning and ex.reasoning:
                lines.append(f"Reasoning: {ex.reasoning}")
            lines.append("")  # blank line between examples
        return "\n".join(lines).rstrip()

    def _item(self, text: str) -> str:
        """Section 3"""
        return (
            "Now label this text:\n"
            f'Text: "{text}"'
        )

    def _format_instruction(self, schema: Schema, reasoning: bool) -> str:
        """Section 4"""
        labels_quoted = ", ".join(f'"{label}"' for label in schema.labels)

        if reasoning:
            template = (
                '{"label": "<label>", "confidence": <0.0–1.0>, '
                '"reasoning": "<brief explanation>"}'
            )
        else:
            template = '{"label": "<label>", "confidence": <0.0–1.0>}'

        return (
            "Respond with ONLY a JSON object in this exact format:\n"
            f"{template}\n\n"
            "Rules:\n"
            f'- "label" must be exactly one of: {labels_quoted}\n'
            '- "confidence" must be a decimal number between 0.0 and 1.0\n'
            "- Do not wrap the JSON in markdown code blocks\n"
            "- Do not include any text before or after the JSON object"
        )

    def _retry_reminder(self, attempt: int) -> str:
        """Section 5 (retry only)"""
        if attempt == 2:
            return (
                "IMPORTANT: Your previous response was not in the required "
                "format. Respond with ONLY the JSON object. No markdown "
                "fences, no explanation, just the JSON."
            )
        return (
            "CRITICAL: You must output ONLY a JSON object. "
            "Start your response with { and end with }. "
            "No other characters before or after."
        )
