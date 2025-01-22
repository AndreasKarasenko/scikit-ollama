from pydantic import BaseModel

from skollama.llm.ollama.clients.ollama.completion import (
    get_chat_completion as _ollama_get_chat_completion,
)


def get_chat_completion(
    messages: dict,
    model: str = "llama3",
    host: str = "http://localhost:11434/",
    options: dict = None,
    format: BaseModel = "",
):
    """Gets a chat completion from an ollama server running locally or remote.

    Parameters
    ----------
    messages : dict
        dictionary with the chat history
    model : str, optional
        model to use, by default "llama3"
    host : str, optional
        Ollama host to connect to, by default "http://localhost:11434/"
    options: dict, optional
        additional options to pass to the Ollama API, by default None
    """
    return _ollama_get_chat_completion(
        messages=messages, model=model, host=host, options=options, format=format
    )
