from skllm.llm.ollama.clients.ollama.credentials import set_credentials
from skllm.utils import retry


@retry(max_retries=3)
def get_embedding(
    text: str,
    model: str = "llama3",
    api: str = "custom_url",
):
    """
    Encodes a string and return the embedding for a string.

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
    max_retries : int, optional
        The maximum number of retries to use. Defaults to 3.
    api: str, optional
        The API to use. Must be one of "openai" or "azure". Defaults to "openai".

    Returns
    -------
    emb : list
        The GPT embedding for the string.
    """
    if api in ("custom_url"):
        client = set_credentials() # should allow for host / url!
    # text = [str(t).replace("\n", " ") for t in text]
    embeddings = []
    emb = client.embeddings(model=model, prompt=text[0]) # change to dont assume multiple texts
    e = emb["embedding"]
    if not isinstance(e, list):
        raise ValueError(
            f"Encountered unknown embedding format. Expected list, got {type(emb)}"
        )
    embeddings.append(e)
    return embeddings
