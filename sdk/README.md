# Tagnify

LLM-powered automated data labeling schema-first, confidence-scored, local-model ready.

```python
from tagnify import Tagnify, Schema, Example

schema = Schema(
    labels=["positive", "negative", "neutral"],
    examples=[Example(text="Great product!", label="positive")]
)

tagnify = Tagnify(model="qwen2.5:7b")
result = tagnify.label("This was a disappointing experience.", schema)

print(result.label)       # "negative"
print(result.confidence)  # 0.91
```

## Features

- **Schema-first design** — labels and few-shot examples are defined upfront; examples are mandatory, not optional
- **Confidence scoring** — every label includes a confidence score; low-confidence results are automatically flagged
- **Automatic retries** — invalid or malformed model output triggers a retry with a stronger prompt, up to 3 attempts
- **Reasoning traces** — optionally request a one-line explanation for each label (`reasoning=True`)
- **Pluggable backends** — run locally for free via Ollama, or plug in your own LLM API

## Installation

```bash
pip install tagnify
```

Requires [Ollama](https://ollama.ai) running locally for the default backend.

## Custom Backends

Have your own LLM API an internal company model, a provider Tagnify doesn't support yet,
anything that isn't Ollama? Implement `BaseBackend` and wire it in with `Tagnify.with_backend()`:

```python
from tagnify import Tagnify, Schema, Example
from tagnify.backends.base import BaseBackend
from tagnify.exceptions import BackendError
import httpx

class MyCompanyBackend(BaseBackend):
    def __init__(self, endpoint: str, api_key: str, model: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str) -> str:
        try:
            response = httpx.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "prompt": prompt},
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"Backend call failed: {e}") from e
        return response.json()["text"]

backend = MyCompanyBackend(endpoint="...", api_key="...", model="...")
tagnify = Tagnify.with_backend(backend)

result = tagnify.label("This was a disappointing experience.", schema)
```

Everything downstream — retries, parsing, validation, confidence scoring — works identically,
regardless of where `complete()` gets its text from. Wrap your own network/API errors in
`BackendError` so the retry logic behaves correctly: it's treated as an infrastructure failure
and is not retried, unlike a malformed model response which is.

## Documentation

Full docs at [docs.tagnify.io](https://docs.tagnify.io) — coming soon.

## License

MIT