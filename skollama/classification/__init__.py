# skollama/classification/__init__.py
"""Text classification models using large language models."""

from skollama.models.ollama.classification.zero_shot import (
    ZeroShotOllamaClassifier,
    MultiLabelZeroShotOllamaClassifier,
    CoTOllamaClassifier,
)

from skollama.models.ollama.classification.few_shot import (
    FewShotOllamaClassifier,
    DynamicFewShotOllamaClassifier,
    MultiLabelFewShotOllamaClassifier,
)

# TODO add OllamaTunableClassifier

__all__ = [
    "ZeroShotOllamaClassifier",
    "MultiLabelZeroShotOllamaClassifier",
    "FewShotOllamaClassifier",
    "DynamicFewShotOllamaClassifier",
    "MultiLabelFewShotOllamaClassifier",
    "CoTOllamaClassifier",
]
