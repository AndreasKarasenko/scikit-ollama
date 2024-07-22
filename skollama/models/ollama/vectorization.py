import warnings

from skllm.models._base.vectorizer import BaseVectorizer as _BaseVectorizer
from skollama.llm.ollama.mixin import OllamaEmbeddingMixin as _OllamaEmbeddingMixin


# TODO refactor to use client instead
class OllamaVectorizer(_BaseVectorizer, _OllamaEmbeddingMixin):
    """An Ollama based embedding model for vectorizing text.

    This vectorizer leverages Ollama served models to create text embeddings.

    Attributes
    ----------
    model : str, optional
        model to use, by default "custom_url::nomic-embed-text"
    batch_size : int, optional
        number of samples per request, by default 1
    url: str, optional
        url to use for the embedding queries, by default localhost

    Warnings
    --------
    OllamaVectorizer is an experimental feature and may undergo significant changes in future releases.
    """

    def __init__(
        self,
        model: str = "custom_url::nomic-embed-text",
        batch_size: int = 1,
        url: str = "http://localhost:11434/api/embeddings",
    ):
        if batch_size > 1:
            warnings.warn("Batch size controlls the number of parallel embedding calls")
        super().__init__(model=model, batch_size=batch_size)
        self.url = url
        self.num_workers = batch_size
        warnings.warn(
            "OllamaVectorizer is an experimental feature and will see some refactoring"
            " in the future."
        )
