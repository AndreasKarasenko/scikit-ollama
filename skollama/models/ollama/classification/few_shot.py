from typing import Optional

from pydantic import BaseModel

from skllm.memory.base import IndexConstructor
from skllm.models._base.classifier import (
    BaseDynamicFewShotClassifier,
    BaseFewShotClassifier,
    MultiLabelMixin,
    SingleLabelMixin,
)
from skllm.models._base.vectorizer import BaseVectorizer
from skollama.llm.ollama.mixin import OllamaClassifierMixin
from skollama.models.ollama.vectorization import OllamaVectorizer


class FewShotOllamaClassifier(
    BaseFewShotClassifier, OllamaClassifierMixin, SingleLabelMixin
):
    """Few-shot text classifier using Ollama API-compatible models.

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
    prompt_template : Optional[str], optional
        custom prompt template to use, by default None
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
        self._base_model = structured_output
        if structured_output and issubclass(structured_output, BaseModel):
            json_schema = structured_output.model_json_schema()
        else:
            json_schema = ""
        self.format = json_schema


class MultiLabelFewShotOllamaClassifier(
    BaseFewShotClassifier, OllamaClassifierMixin, MultiLabelMixin
):
    """Multi-label few-shot text classifier using Ollama API-compatible models.

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
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        options: dict = None,
        default_label: str = "Random",
        max_labels: Optional[int] = 5,
        prompt_template: Optional[str] = None,
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


class DynamicFewShotOllamaClassifier(
    BaseDynamicFewShotClassifier, OllamaClassifierMixin, SingleLabelMixin
):
    """Dynamic few-shot text classifier using Ollama API-compatible models. For
    each sample, N closest examples are retrieved from the memory.

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
    n_examples : int, optional
        number of closest examples per class to be retrieved, by default 3
    memory_index : Optional[IndexConstructor], optional
        custom memory index, for details check `skllm.memory` submodule, by default None
    vectorizer : Optional[BaseVectorizer], optional
        scikit-llm vectorizer; if None, `OllamaVectorizer` is used, by default None
    metric : Optional[str], optional
        metric used for similarity search, by default "euclidean"
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        options: dict = None,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        n_examples: int = 3,
        memory_index: Optional[IndexConstructor] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        metric: Optional[str] = "euclidean",
        **kwargs,
    ):
        if vectorizer is None:
            vectorizer = OllamaVectorizer(model="custom_url::nomic-embed-text")
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            n_examples=n_examples,
            memory_index=memory_index,
            vectorizer=vectorizer,
            metric=metric,
        )
        self.host = host
        self.options = options
