from skllm.models._base.text2text import BaseTranslator
from skollama.llm.ollama.mixin import OllamaCompletionMixin


class OllamaTranslator(
    BaseTranslator,
    OllamaCompletionMixin,
):
    """Text summarizer using Ollama API compatible models.

    Attributes
    ----------
    model : str, optional
        model to use, by default "gpt-3.5-turbo"
    output_language : str, optional
        language to translate to, by default "English"
    host: str, optional
        Ollama host to connect to, by default "http://localhost:11434"
    options: dict, optional
        additional options to pass to the Ollama API, by default None
    """

    def __init__(
        self,
        model: str = "llama3",
        output_language: str = "English",
        host: str = "http://localhost:11434",
        options: dict = None,
    ):
        self.model = model
        self.output_language = output_language
        self.host = host
        self.options = options
