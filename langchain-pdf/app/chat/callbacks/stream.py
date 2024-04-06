from typing import Any
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult


class StreamingHandler(BaseCallbackHandler):
    """Custom callback handler for streaming llm generated tokens"""

    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any
    ) -> Any:
        self.queue.put(token)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any
    ) -> Any:
        self.queue.put(None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any
    ) -> Any:
        self.queue.put(None)
