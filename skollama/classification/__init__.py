# skollama/classification/__init__.py
"""Text classification models using large language models."""

from skollama.models.ollama.classification.zero_shot import (
    ZeroShotOllamaClassifier,
    MultiLabelZeroShotOllamaClassifier,
    # TODO add CoTOllamaClassifier
)

from skollama.models.ollama.classification.few_shot import (
    FewShotOllamaClassifier,
    DynamicFewShotOllamaClassifier,
    # TODO add MultiLabelFewShotOllamaClassifier
)

# TODO add OllamaTunableClassifier

__all__ = [
    "ZeroShotOllamaClassifier",
    "MultiLabelZeroShotOllamaClassifier",
    "FewShotOllamaClassifier",
    "DynamicFewShotOllamaClassifier",
]
