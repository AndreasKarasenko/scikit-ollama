from skllm.llm.gpt.utils import split_to_api_and_model
from skollama.llm.ollama.clients.ollama.embedding import (
    get_embedding as _ollama_get_embedding,
)


def get_embedding(
    text: str,
    model: str = "custom_url::llama3",
):
    """Encodes a string and return the embedding for a string.

    Parameters
    ----------
    text : str
        The string to encode.
    model : str
        model to use, by default "llama3"

    Returns
    -------
    emb : list
        The embedding for the string.
    """
    api, model = split_to_api_and_model(model)
    if api == "gpt4all":
        raise ValueError("GPT4All is not supported for embeddings")
    return _ollama_get_embedding(text, model, api)
