import warnings
from skllm.models._base.vectorizer import BaseVectorizer as _BaseVectorizer
from skollama.llm.ollama.mixin import OllamaEmbeddingMixin as _OllamaEmbeddingMixin
from typing import Optional

# TODO refactor to use client instead
class OllamaVectorizer(_BaseVectorizer, _OllamaEmbeddingMixin):
    def __init__(
        self,
        model: str = "custom_url::nomic-embed-text",
        batch_size: int = 1,
        url: str = "http://localhost:11434/api/embeddings"
    ):
        """
        Text vectorizer using OpenAI/GPT API-compatible models.

        Parameters
        ----------
        model : str, optional
            model to use, by default "text-embedding-ada-002"
        batch_size : int, optional
            number of samples per request, by default 1
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config, by default None
        org : Optional[str], optional
            estimator-specific ORG key; if None, retrieved from the global config, by default None
        """
        if batch_size > 1:
            warnings.warn("Batch size controlls the number of parallel embedding calls")
        super().__init__(model=model, batch_size=batch_size)
        self.url = url
        self.num_workers = batch_size
        warnings.warn("OllamaVectorizer is an experimental feature and will likely be removed at a future point. Support for this feature will only last until /v1/embeddings support is implemented.")
