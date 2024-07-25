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
        output_language: str = "English",
        host: str = "http://localhost:11434",
        options: dict = None,
    ):
        self.model = model
        self.output_language = output_language
        self.host = host
        self.options = options
