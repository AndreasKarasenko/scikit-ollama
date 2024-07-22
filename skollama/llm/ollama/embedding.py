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
    key : str
        The OPEN AI key to use.
    org : str
        The OPEN AI organization ID to use.
    model : str, optional
        The model to use. Defaults to "text-embedding-ada-002".

    Returns
    -------
    emb : list
        The GPT embedding for the string.
    """
    api, model = split_to_api_and_model(model)
    if api == "gpt4all":
        raise ValueError("GPT4All is not supported for embeddings")
    return _ollama_get_embedding(text, model, api)
