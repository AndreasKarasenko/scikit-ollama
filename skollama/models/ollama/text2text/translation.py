from typing import Optional

from pydantic import BaseModel

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
    structured_output : Optional[BaseModel], optional
        structured output model to force output style, by default ""
    """

    def __init__(
        self,
        model: str = "llama3",
        output_language: str = "English",
        host: str = "http://localhost:11434",
        options: dict = None,
        structured_output: Optional[BaseModel] = "",
    ):
        self.model = model
        self.output_language = output_language
        self.host = host
        self.options = options

        self.structured_output = structured_output
        if structured_output and issubclass(structured_output, BaseModel):
            json_schema = structured_output.model_json_schema()
        else:
            json_schema = ""
        self.format = json_schema
