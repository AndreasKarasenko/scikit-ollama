from ollama import Client

def set_credentials(host: str = "http://localhost:11434"):
    """Set the OpenAI key and organization.
    
    Parameters
    ----------
    url : str
        The url for the ollama server.
    model : str
        The model to use.
    """
    client = Client(host=host)
    return client