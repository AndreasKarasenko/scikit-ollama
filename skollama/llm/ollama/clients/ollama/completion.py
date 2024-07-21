from skllm.utils import retry
from ollama import Client

@retry(max_retries=3)
def get_chat_completion(
    messages: dict,
    model: str = "llama3",
    host: str ="http://localhost:11434",
    options: dict = None
):
    """Gets a chat completion from an ollama server.
    
    Parameters
    ----------
    messages : dict
        input messages to use.
    model : str, optional
        model to use, must be available at the server
    host : str, optional
        host url to pass the requests to
    
    Returns
    -------
    completion : dict
    """
    client = Client(host=host)
    # completion = client.generate(model=model, messages=messages) # TODO proliferate options
    completion = client.chat(model=model, messages=messages, options=options) # TODO proliferate options
    return completion