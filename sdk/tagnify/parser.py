#parser from raw to structured data

import json
import re

from tagnify.exceptions import OutputParserError


class OutputParser:
    def parse(self, raw_text: str) -> dict:
        if not raw_text or raw_text.strip():
            raise OutputParserError(
                "LLM returned an empty response."
                "The model may not be loaded yet or the prompt may be too long."
            )
        
        text = raw_text.strip()

        #direct
        result = self._try_parse(text)
        if result is not None:
            return result
        
        #strip code blocks
        cleaned = self._strip_code_blocks(text)
        result = self._try_parse(cleaned)
        if result is not None:
            return result
        
        #extract json from text
        extracted = self._extract_json_from_text(text)
        if extracted:
            result = self._try_parse(extracted)
            if result is not None:
                return result
            
        #fix quote commas
        candidate = extracted or cleaned
        fixed = self._fix_quote_commas(candidate)
        result = self._try_parse(fixed)
        if result is not None:
            return result
        
        #all strategies failed, raise error with preview
        preview = raw_text[:300] + ("..." if len(raw_text) > 300 else "")
        raise OutputParserError(
            f"Could not extract valid JSON from LLM response after all strategies.\n"
            f"Response preview: {preview!r}"
        )

    #parse functions
    def _try_parse(self, text: str) -> dict | None:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return None
        except (json.JSONDecodeError, ValueError):
            return None
        
    def _strip_code_blocks(self, text: str) -> str:
        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
        text = re.sub(r'\s*```\s*$', '', text.strip())
        return text.strip()
    
    def _extract_json_from_text(self, text: str) -> str:
        start = -1
        depth = 0
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i  
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start:i + 1]

        return None  
    
    def _fix_quote_commas(self, text: str) -> str:
        text = re.sub(r"'([^']*)'", r'"\1"', text)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text
