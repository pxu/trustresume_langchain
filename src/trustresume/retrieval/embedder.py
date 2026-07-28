"""Embedding model used to vectorize candidate evidence chunks.

Wraps ``fastembed`` behind LangChain's ``Embeddings`` interface
(``embed_documents``/``embed_query``) so it can be handed to
``langchain_chroma.Chroma`` as its ``embedding_function``.

Milestone M2 (storage + retrieval).
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbeddings(Embeddings):
    """Local, offline embedding model via ``fastembed`` (384-dim by default).

    The underlying model is lazy-loaded on first ``embed_documents``/
    ``embed_query`` call, not at construction — so building a vector store
    never triggers a model download by itself.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        assert self._model is not None
        return [list(map(float, vector)) for vector in self._model.embed(texts)]  # type: ignore[attr-defined]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
