from typing import Optional

from pydantic import BaseModel

from skllm.models._base.classifier import BaseCoTClassifier as _BaseCoTClassifier
from skllm.models._base.classifier import (
    BaseZeroShotClassifier as _BaseZeroShotClassifier,
)
from skllm.models._base.classifier import MultiLabelMixin as _MultiLabelMixin
from skllm.models._base.classifier import SingleLabelMixin as _SingleLabelMixin
from skollama.llm.ollama.mixin import OllamaClassifierMixin as _OllamaClassifierMixin


class ZeroShotOllamaClassifier(
    _BaseZeroShotClassifier, _OllamaClassifierMixin, _SingleLabelMixin
):
    """Zero-shot text classifier using ollama API-compatible models.

    Parameters
    ----------
    model : str, optional
        model to use, by default "llama3"
    host: str, optional
        Ollama host to connect to, by default "http://localhost:11434"
    options: dict, optional
        additional options to pass to the Ollama API, by default None
    default_label : str, optional
        default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
    prompt_template : Optional[str], optional
        custom prompt template to use, by default None
    structured_output : Optional[BaseModel], optional
        structured output model to force output style, by default ""
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        options: dict = None,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        structured_output: Optional[BaseModel] = "",
        **kwargs,
    ):
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            **kwargs,
        )
        self.host = host
        self.options = options
        self.structured_output = structured_output
        if structured_output and issubclass(structured_output, BaseModel):
            json_schema = structured_output.model_json_schema()
        else:
            json_schema = ""
        self.format = json_schema


class CoTOllamaClassifier(
    _BaseCoTClassifier, _OllamaClassifierMixin, _SingleLabelMixin
):
    """Chain-of-though text classifier using Ollama API compatible models.

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
    max_labels : Optional[int], optional
        maximum labels per sample, by default 5
    prompt_template : Optional[str], optional
        custom prompt template to use, by default None
    structured_output : Optional[BaseModel], optional
        structured output model to force output style, by default ""
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        options: dict = None,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        structured_output: Optional[BaseModel] = "",
        **kwargs,
    ):
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            **kwargs,
        )
        self.host = host
        self.options = options
        self.structured_output = structured_output
        if structured_output and issubclass(structured_output, BaseModel):
            json_schema = structured_output.model_json_schema()
        else:
            json_schema = ""
        self.format = json_schema


class MultiLabelZeroShotOllamaClassifier(
    _BaseZeroShotClassifier, _OllamaClassifierMixin, _MultiLabelMixin
):
    """Multi-label zero-shot text classifier using Ollama API-compatible
    models.

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
    max_labels : Optional[int], optional
        maximum labels per sample, by default 5
    prompt_template : Optional[str], optional
        custom prompt template to use, by default None
    structured_output : Optional[BaseModel], optional
        structured output model to force output style, by default ""
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        options: dict = None,
        default_label: str = "Random",
        max_labels: Optional[int] = 5,
        prompt_template: Optional[str] = None,
        structured_output: Optional[BaseModel] = "",
        **kwargs,
    ):
        super().__init__(
            model=model,
            default_label=default_label,
            max_labels=max_labels,
            prompt_template=prompt_template,
            **kwargs,
        )
        self.host = host
        self.options = options

        self.structured_output = structured_output
        if structured_output and issubclass(structured_output, BaseModel):
            json_schema = structured_output.model_json_schema()
        else:
            json_schema = ""
        self.format = json_schema
