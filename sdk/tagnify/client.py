#tagnify client as a public entry for the sdk

from tagnify.backends.base import BaseBackend
from tagnify.backends.ollama import OllamaBackend
from tagnify.engine import LabelEngine
from tagnify.schema import Schema, LabelResult



class Tagnify:
    """Main tagnify client class"""
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        ollama_host: str = "http://localhost:11434",
        max_retries: int = 3,
        timeout: float = 120.0,
        temperature: float = 0.1,
    ) -> None:
        self._backend = self._init_backend(
            model=model,
            api_key=api_key,
            ollama_host=ollama_host,
            timeout=timeout,
            temperature=temperature,
        )
        self._engine = LabelEngine(
            backend=self._backend,
            max_retries=max_retries,
        )

    @classmethod
    def with_backend(
        cls,
        backend: BaseBackend,
        max_retries: int = 3,
    ) -> "Tagnify":
        """Create a Tagnify client using a custom backend"""
        if not isinstance(backend, BaseBackend):
            raise TypeError(
                f"with_backend() requires a BaseBackend instance, got "
                f"{type(backend).__name__}. Subclass BaseBackend and "
                f"implement complete(prompt: str) -> str to create a "
                f"custom backend."
            )

        instance = cls.__new__(cls)
        instance._backend = backend
        instance._engine = LabelEngine(backend=backend, max_retries=max_retries)
        return instance

    def _init_backend(
        self,
        model: str,
        api_key: str | None,
        ollama_host: str,
        timeout: float,
        temperature: float,
    ) -> BaseBackend:
        """Select and initialize backend based on api_key, None = free tier, set = GroqBackend"""

        if api_key is None:
            return OllamaBackend(
                model=model,
                host=ollama_host,
                timeout=timeout,
                temperature=temperature,
            )
        
        raise NotImplementedError(
            "Cloud backend support (api_key) is coming soon. "
            "For now, use Tagnify without api_key to run locally with Ollama. "
             "or Tagnify.with_backend() to use your own LLM API. "
        )
    
    def label(
        self,
        text: str,
        schema: Schema,
        reasoning: bool = False,
    ) -> LabelResult:
        """Label a single text item"""
        return self._engine.run(text=text, schema=schema, reasoning=reasoning)
    
    def label_batch(
        self,
        texts: list[str],
        schema: Schema,
        reasoning: bool = False,
    ) -> list[LabelResult]:
        """label list of text items"""
        return [
            self.label(text=text, schema=schema, reasoning=reasoning)
            for text in texts
        ]
