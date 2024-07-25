from typing import Optional

from skllm.models._base.text2text import BaseSummarizer
from skollama.llm.ollama.mixin import OllamaCompletionMixin


class OllamaSummarizer(
    BaseSummarizer,
    OllamaCompletionMixin,
):
    """Text summarizer using Ollama API compatible models.

    Attributes
    ----------
    model : str, optional
        model to use, by default "gpt-3.5-turbo"
    default_label : str, optional
        default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
    max_labels : Optional[int], optional
        maximum labels per sample, by default 5
    prompt_template : Optional[str], optional
        custom prompt template to use, by default None
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        options: dict = None,
        max_words: int = 15,
        focus: Optional[str] = None,
    ):
        self.model = model
        self.max_words = max_words
        self.focus = focus
        self.host = host
        self.options = options
