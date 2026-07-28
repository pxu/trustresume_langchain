"""Deterministic test doubles shared across the test suite.

Kept in a standalone module (not conftest) so tests can also construct these
directly and ``isinstance``-check them against ``langchain_core.embeddings.Embeddings``.
"""

from __future__ import annotations

import hashlib

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import Runnable


class FakeEmbeddings(Embeddings):
    """Deterministic hash-based embedder for tests.

    Same text always embeds identically; no ML model, no network. Mirrors
    ``langchain_core.embeddings.Embeddings``'s two-method interface
    (``embed_documents``/``embed_query``) rather than the original's single
    ``embed`` method.
    """

    def __init__(self, dimension: int = 16) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class FakeToolCallingChatModel(GenericFakeChatModel):
    """A ``GenericFakeChatModel`` that also supports ``with_structured_output``.

    ``GenericFakeChatModel.bind_tools`` raises ``NotImplementedError`` (the
    base ``BaseChatModel`` default), which is what ``with_structured_output``
    relies on — so structured-output agents can't be tested against the
    stock fake. Overriding ``bind_tools`` to just return ``self`` is enough:
    the fake doesn't need to honor the bound tool schema, since its scripted
    ``AIMessage`` already carries the exact ``tool_calls`` the test wants
    validated into the target Pydantic model.
    """

    def bind_tools(self, tools: object, **kwargs: object) -> Runnable:
        return self


def scripted_tool_call(name: str, args: dict[str, object]) -> FakeToolCallingChatModel:
    """A ``FakeToolCallingChatModel`` that returns one scripted tool call.

    The LangChain equivalent of pydantic-ai's
    ``TestModel(custom_output_args={...})`` — ``name`` must match the target
    Pydantic model's class name (that's the tool name ``with_structured_output``
    looks for), and ``args`` must match its fields.
    """
    return FakeToolCallingChatModel(
        messages=iter(
            [AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id="call_1")])]
        )
    )
