# Scikit-Ollama: an extension of Scikit-LLM for Ollama served models.

Leverage the power of Scikit-LLM and the security of self-hosted LLMs.

## Installation

```bash
pip install scikit-ollama
```

## Support us

You can support the project in the following ways:

- Support the original [Scikit-LLM](https://github.com/iryna-kondr/scikit-llm) package. New features will be made available downstream slowly but surely.
- Star this repository.
- Provide feedback in the [issues](<>) section.
- Share this repository with others.

## Quick start & documentation

Assuming you have installed and configured Ollama to run on your machine:

```python
from skllm.datasets import get_classification_dataset
from skollama.models.ollama.classification.zero_shot import ZeroShotOllamaClassifier

X, y = get_classification_dataset()

clf = ZeroShotOllamaClassifier(model="llama3:8b")
clf.fit(X, y)
preds = clf.predict(X)
```

For more information please refer to the [documentation](<>).

## Why Scikit-Ollama?

Scikit-Ollama lets you use locally run models for several text classification approaches.
Running models locally can be beneficial for cases where data privacy and control are paramount. This also makes you less dependent on 3rd-party APIs and gives you more control over when you want to add changes.

This project builds heavily on Scikit-LLM and has it as a core dependency. Scikit-LLM provides broad and great support to query a variety of backend families,
e.g. OpenAI, Vertex, GPT4All. In their version you could already use the OpenAI compatible v1 API backend to query locally run models. However, the issue is that Ollama does not support passing options, such as the context size to that endpoint.

Therefore this model uses the Ollama Python SDK to allow that level of control.

## Citation

```
@software{Scikit-Ollama,
    author = {Andreas Karasenko},
    year = {2024},
    title = {Scikit-Ollama: an extension of Scikit-LLM for Ollama served models},
    url = {https://github.com/AndreasKarasenko/scikit-ollama}
}
```
