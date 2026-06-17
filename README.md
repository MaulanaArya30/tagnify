# Tagnify

**LLM-powered automated data labeling. Schema-first. Confidence-scored. Local-model ready.**

[![Tests](https://github.com/MaulanaArya30/tagnify/actions/workflows/test.yml/badge.svg)](https://github.com/MaulanaArya30/tagnify/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/tagnify/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

```python
from tagnify import Tagnify, Schema, Example

schema = Schema(
    labels=["positive", "negative", "neutral"],
    examples=[Example(text="Arrived hot and on time!", label="positive")]
)

tagnify = Tagnify(model="qwen2.5:7b")
result = tagnify.label("The packaging was damaged and food was cold.", schema)

print(result.label)       # "negative"
print(result.confidence)  # 0.91
print(result.flagged)     # False
```

---

## What is Tagnify?

Tagnify is a Python SDK for automated data labeling using local LLMs. You define a schema — what labels are valid, what examples look like — and Tagnify handles everything else: prompt construction, LLM communication, output parsing, validation, retry logic, and confidence scoring.

It runs entirely on your machine using [Ollama](https://ollama.ai). No API keys. No data leaving your machine. No per-label charges for local use.

---

## Why Tagnify?

### The problem with raw LLM labeling

Calling an LLM directly for labeling seems simple until production. Then you hit:

- LLM returns `"Positive"` when your label is `"positive"` — silent mismatch
- Model outputs JSON wrapped in markdown fences — parse error
- Model returns `"I think this is positive"` — not JSON at all
- Low-confidence edge cases go straight into your dataset — noise
- One bad item crashes your entire batch — data loss

### What Tagnify does differently

**Schema-first design with mandatory examples.** You don't just tell Tagnify what labels exist — you show it what a correct label looks like. Few-shot examples are required, not optional. This is the single biggest cause of wrong label fields in production LLM pipelines.

**Structured retry logic.** On bad output, Tagnify retries with a progressively stronger prompt reminder. Attempt 2 adds a format correction. Attempt 3 adds explicit character-level instructions. The retry loop is in one place and always returns a result — it never crashes your batch.

**Confidence scoring and flagging.** Every label comes with a confidence score. Results below your schema's threshold are automatically flagged for human review. You decide the threshold per schema.

**Clean output parsing.** The `OutputParser` handles markdown fences, single quotes, trailing commas, surrounding text, and nested objects — the full range of what models actually return in the real world.

**Pluggable backends.** Ollama isn't the only option. If you have your own LLM API — an internal company model, a provider Tagnify doesn't support yet — implement one method and plug it straight in. The retry logic, parsing, validation, and confidence scoring all work unchanged regardless of where the model's text comes from.

---

## Installation

```bash
pip install tagnify
```

Tagnify requires [Ollama](https://ollama.ai) running locally.

**Install Ollama:**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Pull a model:**

```bash
ollama pull qwen2.5:7b       # recommended — fast, accurate, 4.7GB
ollama pull deepseek-r1:8b   # strong reasoning, 4.9GB
ollama pull llama3.2:3b      # lightweight option, 2.0GB
```

**Start Ollama:**

```bash
ollama serve
```

That's it. No API keys. No accounts.

---

## Quick start

### Sentiment classification

```python
from tagnify import Tagnify, Schema, Example

schema = Schema(
    labels=["positive", "negative", "neutral"],
    description="Customer reviews for a food delivery app",
    examples=[
        Example(text="Arrived hot and on time, driver was great!", label="positive"),
        Example(text="Food was cold and an hour late.", label="negative"),
        Example(text="Order was fine, nothing special.", label="neutral"),
    ],
    confidence_threshold=0.7,  # flag anything below 70% confidence
)

tagnify = Tagnify(model="qwen2.5:7b")

result = tagnify.label("The app kept crashing during checkout.", schema)
print(result.label)       # "negative"
print(result.confidence)  # 0.88
print(result.flagged)     # False — above 0.7 threshold
print(result.attempts)    # 1 — succeeded first try
```

### Batch labeling

```python
reviews = [
    "Best delivery I've ever had.",
    "App is okay but nothing impressive.",
    "Driver cancelled without notice.",
    "Arrived early with a free drink!",
]

results = tagnify.label_batch(reviews, schema)

for text, result in zip(reviews, results):
    status = "⚠ flagged" if result.flagged else "✓"
    print(f"{status} [{result.label}] ({result.confidence:.2f}) — {text[:40]}")
```

### With reasoning traces

```python
result = tagnify.label(
    "Packaging was a bit dented but food was still warm.",
    schema,
    reasoning=True,
)

print(result.label)     # "neutral"
print(result.reasoning) # "Mixed signals: packaging complaint offset by warm food."
```

### Handling failures gracefully

```python
results = tagnify.label_batch(large_dataset, schema)

successes = [r for r in results if r.success]
failures  = [r for r in results if not r.success]
flagged   = [r for r in results if r.flagged and r.success]

print(f"Labeled:  {len(successes)}")
print(f"Failed:   {len(failures)}")   # will never crash the batch
print(f"Flagged:  {len(flagged)}")    # needs human review
```

### Custom backends (bring your own LLM)

Have your own LLM API — an internal company model, a provider Tagnify doesn't support yet, anything that isn't Ollama? Implement `BaseBackend` and wire it in with `Tagnify.with_backend()`:

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

Everything downstream — retries, parsing, validation, confidence scoring, flagging — works identically to the built-in backends, because all of it operates on the string `complete()` returns, regardless of where that string came from. Wrap your own network/API errors in `BackendError` so the retry logic behaves correctly: it's treated as an infrastructure failure and is not retried, unlike a malformed model response, which is.

---

## API reference

### `Schema`

The labeling contract. Defines what labels are valid, what examples teach the model, and how to score confidence.

```python
Schema(
    labels=["positive", "negative"],   # required — minimum 2, must be unique
    examples=[...],                    # required — minimum 1, non-negotiable
    description=None,                  # optional — task context injected into prompt
    confidence_threshold=0.5,          # optional — below this → result.flagged=True
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `labels` | `list[str]` | required | Valid label values. Minimum 2, must be unique. |
| `examples` | `list[Example]` | required | Few-shot examples. Minimum 1. |
| `description` | `str \| None` | `None` | Task-level context injected into the prompt. |
| `confidence_threshold` | `float` | `0.5` | Results below this value get `flagged=True`. |

**Validation at instantiation time** — schema errors are caught before any LLM call:

```python
# Raises immediately — not buried in a pipeline failure
Schema(labels=["positive"], examples=[...])
# ValueError: Schema must define at least 2 labels

Schema(labels=["positive", "negative"], examples=[])
# ValueError: Schema requires at least one example — this is non-negotiable

Schema(
    labels=["positive", "negative"],
    examples=[Example(text="Fine.", label="neutral")]  # "neutral" not in labels
)
# ValueError: Example labels ['neutral'] are not defined in the labels list
```

**Dict escape hatch** — same validation, different syntax:

```python
schema = Schema.from_dict({
    "labels": ["positive", "negative"],
    "examples": [{"text": "Great!", "label": "positive"}],
    "confidence_threshold": 0.8,
})
```

---

### `Example`

A single few-shot example that teaches the model what a correct label looks like.

```python
Example(
    text="The delivery was perfect.",  # required
    label="positive",                  # required — must be in schema.labels
    reasoning=None,                    # optional — chain-of-thought hint
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | required | The input text being labeled. |
| `label` | `str` | required | The correct label for this text. |
| `reasoning` | `str \| None` | `None` | Why this label is correct. Injected as chain-of-thought context when `reasoning=True`. |

---

### `Tagnify`

The main client. Instantiate once, call `label()` many times.

```python
Tagnify(
    model="qwen2.5:7b",              # required — Ollama model name with tag
    api_key=None,                     # None = local Ollama; set = cloud (coming soon)
    ollama_host="http://localhost:11434",
    max_retries=3,
    timeout=120.0,
    temperature=0.1,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Ollama model name. Must be pulled first: `ollama pull qwen2.5:7b`. |
| `api_key` | `str \| None` | `None` | Cloud API key. `None` uses local Ollama. |
| `ollama_host` | `str` | `http://localhost:11434` | Override for Docker or remote Ollama. |
| `max_retries` | `int` | `3` | Max labeling attempts per item before returning a failure result. |
| `timeout` | `float` | `120.0` | Seconds to wait for model response. Local models can be slow on first load. |
| `temperature` | `float` | `0.1` | Sampling temperature. Lower = more deterministic. Recommended: 0.1 for classification. |

**Methods:**

```python
# Label a single item
result: LabelResult = tagnify.label(text, schema, reasoning=False)

# Label a list of items — failures don't stop the batch
results: list[LabelResult] = tagnify.label_batch(texts, schema, reasoning=False)
```

**Alternate constructor — custom backends:**

```python
# Skip Ollama and the cloud API entirely — use any BaseBackend implementation
tagnify = Tagnify.with_backend(my_backend, max_retries=3)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | `BaseBackend` | required | Any class implementing `complete(prompt: str) -> str`. |
| `max_retries` | `int` | `3` | Same meaning as on the standard constructor. |

Raises `TypeError` immediately if `backend` isn't a `BaseBackend` instance, rather than failing confusingly later inside the labeling pipeline.

---

### `LabelResult`

The output of every label call. Always returned — never raises on labeling failures.

```python
@dataclass
LabelResult:
    label: str | None      # None only when success=False
    confidence: float      # 0.0 – 1.0
    reasoning: str | None  # populated when reasoning=True
    flagged: bool          # True when confidence < schema.confidence_threshold
    attempts: int          # how many tries the engine needed (1–max_retries)
    success: bool          # False when all retries exhausted
    error: str | None      # readable explanation on failure
```

| Field | Description |
|---|---|
| `label` | The assigned label. `None` only on complete failure. |
| `confidence` | Model's confidence in the label. 0.0–1.0. |
| `reasoning` | Model's explanation. Only populated when `reasoning=True`. |
| `flagged` | `True` when `confidence < schema.confidence_threshold`. Also `True` on failure. |
| `attempts` | Number of tries needed. A result taking 3 attempts warrants logging. |
| `success` | `False` when all retries are exhausted. |
| `error` | Describes the last failure when `success=False`. |

---

## How confidence scoring works

Every label call asks the model to return a confidence score alongside its label. Tagnify uses this score to automatically flag uncertain results:

```python
schema = Schema(
    labels=["positive", "negative", "neutral"],
    examples=[...],
    confidence_threshold=0.75,  # flag anything below 75%
)

result = tagnify.label("This was okay I guess.", schema)
# result.confidence = 0.61
# result.flagged    = True  ← below 0.75 threshold
# result.label      = "neutral"  ← label is still present, just flagged
# result.success    = True        ← this is not a failure
```

Flagged results have a label — they're uncertain, not wrong. The typical workflow is to send flagged items to a human review queue rather than discarding them.

A `result.flagged = True` with `result.success = True` means the model labeled it but wasn't confident. A `result.flagged = True` with `result.success = False` means all retries failed — there is no label.

---

## How retry logic works

When the model returns malformed or invalid output, Tagnify retries automatically:

```
Attempt 1: normal prompt
Attempt 2: adds "IMPORTANT: your last response was wrong format..."
Attempt 3: adds "CRITICAL: start your response with { and end with }..."
```

Each retry passes `attempt=N` to the prompt builder, which escalates the format instruction. If all attempts fail, you get `LabelResult(success=False)` — the batch continues.

The retry loop catches two specific failures: `OutputParseError` (can't extract JSON at all) and `ValidationError` (got JSON but the label isn't valid). Infrastructure failures (`BackendError` — Ollama not running) propagate immediately since retrying won't help.

---

## Output parsing

Tagnify's `OutputParser` handles the full range of what models actually return, not just the ideal case:

| Model output | Strategy used |
|---|---|
| `{"label": "positive", "confidence": 0.9}` | Direct parse |
| ` ```json\n{"label": "positive", ...}\n``` ` | Strip markdown fences |
| `Based on the text: {"label": "positive", ...}` | Extract JSON block |
| `{'label': 'positive', 'confidence': 0.9,}` | Fix single quotes + trailing comma |
| `I think this is positive` | All strategies fail → retry |

---

## Model recommendations

| Model | Size | Speed | Accuracy | Notes |
|---|---|---|---|---|
| `qwen2.5:7b` | 4.7 GB | Fast | High | **Recommended default** |
| `qwen2.5:14b` | 9.0 GB | Medium | Very high | Better for nuanced tasks |
| `deepseek-r1:8b` | 4.9 GB | Medium | High | Strong reasoning traces |
| `llama3.2:3b` | 2.0 GB | Very fast | Medium | Good for high-volume, simpler tasks |

Pull any model with `ollama pull <model-name>` before using it with Tagnify.

---

## Roadmap

### Current — v0.1.x (SDK, local models)
- [x] Schema-first design with Pydantic validation
- [x] OllamaBackend — local models, zero cost
- [x] Confidence scoring and automatic flagging
- [x] Reasoning traces (optional)
- [x] Four-strategy output parsing cascade
- [x] Retry logic with progressive prompt escalation
- [x] Batch labeling
- [x] `Tagnify.with_backend()` — plug in your own LLM API (internal models, unsupported providers)
- [x] Full test suite (90+ tests)

### Phase 2 — Cloud API
- [ ] `GroqBackend` — cloud inference via Groq API
- [ ] `TogetherBackend` — cloud inference via Together AI
- [ ] REST API (FastAPI) — for non-Python users
- [ ] Per-label usage tracking and billing
- [ ] API key management

The upgrade path from local to cloud is a single parameter:

```python
# Local (free)
tagnify = Tagnify(model="qwen2.5:7b")

# Cloud (coming soon — same API, same schemas, same results)
tagnify = Tagnify(model="qwen2.5-7b", api_key="tgnf-...")
```

### Phase 3 — Dashboard and teams
- [ ] Web dashboard — job history, flagged item review queue
- [ ] Team access and shared schemas
- [ ] Webhook integrations
- [ ] Usage analytics

---

## Repository structure

```
tagnify/
├── sdk/                          # Python SDK (pip install tagnify)
│   ├── tagnify/
│   │   ├── __init__.py           # public API surface
│   │   ├── client.py             # Tagnify class
│   │   ├── schema.py             # Schema, Example, LabelResult
│   │   ├── engine.py             # LabelingEngine — retry orchestrator
│   │   ├── prompt.py             # PromptBuilder
│   │   ├── parser.py             # OutputParser — 4-strategy cascade
│   │   ├── validator.py          # Validator
│   │   ├── exceptions.py         # TagnifyError hierarchy
│   │   └── backends/
│   │       ├── base.py           # BaseBackend interface
│   │       └── ollama.py         # OllamaBackend (MVP)
│   └── tests/                    # 90+ tests, all passing
├── api/                          # FastAPI cloud backend (Phase 2)
├── dashboard/                    # Next.js dashboard (Phase 3)
└── .github/workflows/            # CI (test on push) + CD (publish on tag)
```

---

## Development

```bash
git clone https://github.com/MaulanaArya30/tagnify.git
cd tagnify/sdk

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the full test suite
pytest tests/ -v
```

All 90+ tests run without Ollama — backends are mocked in tests.

---

## Contributing

Issues and pull requests are welcome. For significant changes, open an issue first to discuss the direction.

```bash
# Run tests before submitting
pytest tests/ -v

# All tests must pass on Python 3.10, 3.11, and 3.12
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

MIT — see [LICENSE](LICENSE).

---

## About

Tagnify was built by [Maulana Arya Alambana](https://github.com/MaulanaArya30). The retry logic, mandatory few-shot examples, and confidence flagging were extracted from a real LLM labeling pipeline built in production — this isn't a greenfield project.

*Production experience → open-source tool → commercial product. That's the arc.*