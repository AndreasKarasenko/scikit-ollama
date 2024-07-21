from typing import List, Union, Dict, Optional, Any, Mapping
from skllm.llm.ollama.completion import get_chat_completion
from skllm.utils import extract_json_key
from concurrent.futures import ThreadPoolExecutor
from skllm.llm.ollama.embedding import get_embedding
from skllm.llm.base import (
    BaseClassifierMixin,
    BaseEmbeddingMixin,
    BaseTextCompletionMixin,
)
import numpy as np
from tqdm import tqdm
from itertools import repeat


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
    def _get_chat_completion(
        self,
        model: str,
        messages: Union[str, List[Dict[str, str]]],
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
        )
        return completion
    
    def _convert_completion_to_str(self, completion: Mapping[str, Any]):
        if hasattr(completion, "__getitem__"):
            return str(completion["message"]["content"])
        return str(completion.message.content)

class OllamaClassifierMixin(OllamaCompletionMixin, BaseClassifierMixin):
    
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
        try:
            if hasattr(completion, "__getitem__"):
                label = extract_json_key(
                    completion["message"]["content"], "label"
                )
            else:
                label = extract_json_key(completion.message.content, "label")
        except Exception as e:
            print(completion)
            print(f"Could not extract the label from the completion: {str(e)}")
            label = ""
        return label

class OllamaEmbeddingMixin(BaseEmbeddingMixin):
    def _get_embeddings(self, text: np.ndarray) -> List[List[float]]:
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
        print("Batch size:", self.batch_size) # does not work yet, needs refactor of probably WAY more things
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            embs = list(
                tqdm(executor.map(lambda x, y: get_embedding(x,y), text, repeat(self.model, len(text))), total=len(text))
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