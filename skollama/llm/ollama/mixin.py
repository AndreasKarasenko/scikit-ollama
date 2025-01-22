from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import Any, Optional, Union

import numpy as np
from tqdm import tqdm

from skllm.llm.base import (
    BaseClassifierMixin,
    BaseEmbeddingMixin,
    BaseTextCompletionMixin,
)
from skllm.utils import extract_json_key
from skollama.llm.ollama.completion import get_chat_completion
from skollama.llm.ollama.embedding import get_embedding


def construct_message(role: str, content: str) -> dict:
    """Constructs a message for the OpenAI API.

    Parameters
    ----------
    role : str
        The role of the message. Must be one of "system", "user", or "assistant".
    content : str
        The content of the message.

    Returns
    -------
    message : dict
    """
    if role not in ("system", "user", "assistant"):
        raise ValueError("Invalid role")
    return {"role": role, "content": content}


class OllamaCompletionMixin(BaseTextCompletionMixin):
    """Mixin for handling chat completions with the Ollama server.

    This mixin provides methods to interact with the Ollama server for obtaining chat completions.
    It defines the following functionality:

    Methods
    -------
    _get_chat_completion(model, messages, system_message=None, **kwargs)
        Gets a chat completion from the Ollama server based on the provided messages and model.

    _convert_completion_to_str(completion)
        Converts the completion object returned by the Ollama server into a string.
    """

    def _get_chat_completion(
        self,
        model: str,
        messages: Union[str, list[dict[str, str]]],
        system_message: Optional[str] = None,
        **kwargs: Any,
    ):
        """Gets a chat completion from the Ollama server.

        Parameters
        ----------
        model : str
            The model to use.
        messages : Union[str, List[Dict[str, str]]]
            input messages to use.
        system_message : Optional[str]
            A system message to use.
        **kwargs : Any
            placeholder.

        Returns
        -------
        completion : dict
        """
        msgs = []
        if system_message is not None:
            msgs.append(construct_message("system", system_message))
        if isinstance(messages, str):
            msgs.append(construct_message("user", messages))
        else:
            for message in messages:
                msgs.append(construct_message(message["role"], message["content"]))
        completion = get_chat_completion(
            msgs,
            model,
            self.host,
            self.options,
            self.format,
        )
        return completion

    def _convert_completion_to_str(self, completion: Mapping[str, Any]):
        if hasattr(completion, "__getitem__"):
            return str(completion["message"]["content"])
        return str(completion.message.content)


class OllamaClassifierMixin(OllamaCompletionMixin, BaseClassifierMixin):
    """Mixin for Ollama-based classification tasks.

    This mixin extends the OllamaCompletionMixin to provide functionality
    specific to classification tasks, leveraging the Ollama model's ability
    to classify text based on provided inputs and completions.

    Methods
    -------
    _extract_out_label(completion, **kwargs)
        Extracts the classification label from the Ollama model's completion.
    """

    def _extract_out_label(self, completion: Mapping[str, Any], **kwargs) -> Any:
        """Extracts the label from a completion.

        Parameters
        ----------
        label : Mapping[str, Any]
            The label to extract.

        Returns
        -------
        label : str
        """
        if self._base_model:
            key = list(self._base_model.__fields__.keys())[0]
        else:
            key = "label"
        try:
            if hasattr(completion, "__getitem__"):
                label = extract_json_key(completion["message"]["content"], key)
            else:
                label = extract_json_key(completion.message.content, "label")
        except Exception as e:
            print(completion)
            print(f"Could not extract the label from the completion: {str(e)}")
            label = ""
        return label


class OllamaEmbeddingMixin(BaseEmbeddingMixin):
    """Mixin for Ollama-based embedding tasks.

    This mixin extends the BaseEmbeddingMixin to provide a multi-threaded approach
    for getting embeddings.
    It defines the following methods:

    Methods
    -------
    _get_embeddings(self, text: np.ndarray)
        Queries the locally running embedding server with the provided texts.
    """

    def _get_embeddings(self, text: np.ndarray) -> list[list[float]]:
        """Gets embeddings from the OpenAI compatible API.

        Parameters
        ----------
        text : str
            The text to embed.
        model : str
            The model to use.
        batch_size : int, optional
            The batch size to use. Defaults to 1.

        Returns
        -------
        embedding : List[List[float]]
        """
        embeddings = []
        print(
            "Batch size:", self.batch_size
        )  # does not work yet, needs refactor of probably WAY more things
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            embs = list(
                tqdm(
                    executor.map(
                        lambda x, y: get_embedding(x, y),
                        text,
                        repeat(self.model, len(text)),
                    ),
                    total=len(text),
                )
            )
        for i in embs:
            embeddings.extend(i)
        # for i in tqdm(range(0, len(text), self.batch_size)):
        # batch = text[i : i + self.batch_size].tolist()
        # embeddings.extend( # technically a single instance, can be multiprocessed to allow for batches
        #     get_embedding(
        #         batch,
        #         self.model,
        #     )
        # )

        return embeddings
