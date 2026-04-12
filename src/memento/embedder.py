"""Embedding protocol and implementations for Memento."""

from __future__ import annotations

import logging
import os
from typing import Protocol

import numpy as np

from memento.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol for text embedding providers."""

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...


class SentenceTransformerEmbedder:
    """Embedder using sentence-transformers (local, no API calls)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install memento-memory[local-embeddings]"
            )
        logger.info("Loading sentence-transformer model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [embeddings[i] for i in range(len(texts))]


class OpenAIEmbedder:
    """Embedder using OpenAI's embedding API (or any OpenAI-compatible endpoint)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        dimension: int = 384,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for API embeddings. "
                "Install with: pip install memento-memory[openai]"
            )
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            input=text, model=self._model, dimensions=self._dimension
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        response = self._client.embeddings.create(
            input=texts, model=self._model, dimensions=self._dimension
        )
        return [
            np.array(item.embedding, dtype=np.float32)
            for item in sorted(response.data, key=lambda x: x.index)
        ]


class GeminiEmbedder:
    """Embedder using Google's Gemini embedding API."""

    def __init__(
        self,
        model: str = "text-embedding-004",
        api_key: str | None = None,
        dimension: int = 384,
    ) -> None:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai is required for Gemini embeddings. "
                "Install with: pip install memento-memory[gemini]"
            )
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config={"output_dimensionality": self._dimension},
        )
        return np.array(result.embeddings[0].values, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Factory function to create the configured embedder.

    If provider is "auto" (the default), tries sentence-transformers first
    (free, local), then falls back to whichever API key is available.
    """
    if config.provider == "auto":
        # Try local first (free, no API calls)
        try:
            return SentenceTransformerEmbedder(model_name=config.model)
        except ImportError:
            pass

        # Fall back to whichever API key is set
        api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            logger.info("Using OpenAI embeddings (sentence-transformers not installed)")
            return OpenAIEmbedder(api_key=api_key)

        if os.environ.get("GOOGLE_API_KEY"):
            logger.info("Using Gemini embeddings (sentence-transformers not installed)")
            return GeminiEmbedder()

        # Ollama via OpenAI-compatible API
        base_url = os.environ.get("MEMENTO_LLM_BASE_URL")
        provider = os.environ.get("MEMENTO_LLM_PROVIDER", "")
        if provider == "ollama" or (base_url and "localhost" in base_url):
            logger.info("Using Ollama embeddings (sentence-transformers not installed)")
            return OpenAIEmbedder(
                model="nomic-embed-text",
                base_url=base_url or "http://localhost:11434/v1",
                api_key="ollama",
                dimension=384,
            )

        raise ImportError(
            "No embedding provider available. Either:\n"
            "  pip install memento-memory[local-embeddings]  (local, no API calls)\n"
            "  Set OPENAI_API_KEY  (OpenAI embeddings)\n"
            "  Set GOOGLE_API_KEY  (Gemini embeddings)\n"
            "  Set MEMENTO_LLM_PROVIDER=ollama  (Ollama embeddings)"
        )

    if config.provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=config.model)
    elif config.provider == "openai":
        return OpenAIEmbedder(
            model=config.model,
            api_key=config.openai_api_key,
            base_url=os.environ.get("MEMENTO_EMBEDDING_BASE_URL"),
        )
    elif config.provider == "gemini":
        return GeminiEmbedder(model=config.model)
    elif config.provider in ("ollama", "openai-compatible"):
        base_url = os.environ.get("MEMENTO_EMBEDDING_BASE_URL") or os.environ.get("MEMENTO_LLM_BASE_URL") or "http://localhost:11434/v1"
        return OpenAIEmbedder(
            model=config.model or "nomic-embed-text",
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY") or "ollama",
            dimension=config.dimension,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")
