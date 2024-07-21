import warnings
from skllm.llm.ollama.clients.ollama.completion import (
    get_chat_completion as _ollama_get_chat_completion
)


def get_chat_completion(
    messages: dict,
    model: str = "llama3",
    host: str = "http://localhost:11434/",
    options: dict = None,
):
    """Gets a chat completion from an ollama server running locally or remote"""
    return _ollama_get_chat_completion(
        messages=messages,
        model=model,
        host=host,
        options=options
    )