from typing import Optional

from pydantic import BaseModel

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
        model to use, by default "llama3"
    host: str, optional
        Ollama host to connect to, by default "http://localhost:11434"
    options: dict, optional
        additional options to pass to the Ollama API, by default None
    default_label : str, optional
        default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
    max_words : Optional[int], optional
        maximum number of words to use, by default 15
    structured_output : Optional[BaseModel], optional
        structured output model to force output style, by default ""
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        options: dict = None,
        max_words: int = 15,
        focus: Optional[str] = None,
        structured_output: Optional[BaseModel] = "",
    ):
        self.model = model
        self.max_words = max_words
        self.focus = focus
        self.host = host
        self.options = options

        self.structured_output = structured_output
        if structured_output and issubclass(structured_output, BaseModel):
            json_schema = structured_output.model_json_schema()
        else:
            json_schema = ""
        self.format = json_schema
