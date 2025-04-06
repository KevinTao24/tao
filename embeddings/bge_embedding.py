from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers.SentenceTransformer import SentenceTransformer

DEFAULT_BGE_MODEL = "BAAI/bge-large-zh-v1.5"
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："


class PABgeEmbeddings(BaseModel, Embeddings):
    """Bge embedding models.

    Bge Example:
        .. code-block:: python

            from tao.embeddings import PABgeEmbeddings

            model_name = "BAAI/bge-large-zh-v1.5"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            bge = PABgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any = None  #: :meta private:
    model_name: str = DEFAULT_BGE_MODEL
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models.
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION_ZH
    """Instruction to use for embedding query."""
    embed_instruction: str = ""
    """Instruction to use for embedding document."""
    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH
        self.client = SentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [self.embed_instruction + t.replace("\n", " ") for t in texts]
        embeddings = self.client.encode(
            texts, show_progress_bar=self.show_progress, **self.encode_kwargs
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.client.encode(
            self.query_instruction + text,
            show_progress_bar=self.show_progress,
            **self.encode_kwargs,
        )
        return embedding.tolist()
