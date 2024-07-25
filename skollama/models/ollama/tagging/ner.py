from skllm.models._base.tagger import ExplainableNER
from skollama.llm.ollama.mixin import OllamaCompletionMixin


class OllamaExplainableNER(ExplainableNER, OllamaCompletionMixin):
    """Named entity recognition using Ollama API compatible models.

    Attributes
    ----------
    entities : dict
        dictionary of entities to recognize, with keys as entity names and values as descriptions
    display_predictions : bool, optional
        whether to display predictions, by default False
    sparse_output : bool, optional
        whether to generate a sparse representation of the predictions, by default True
    model : str, optional
        model to use, by default "gpt-3.5-turbo"
    """

    def __init__(
        self,
        entities: dict[str, str],
        display_predictions: bool = False,
        sparse_output: bool = True,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        options: dict = None,
        num_workers: int = 1,
    ):
        self.model = model
        self.entities = entities
        self.display_predictions = display_predictions
        self.sparse_output = sparse_output
        self.host = host
        self.options = options
        self.num_workers = num_workers
