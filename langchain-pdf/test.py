from typing import Any
from uuid import UUID
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
from queue import Queue
from threading import Thread

load_dotenv()


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


chat = ChatOpenAI(streaming=True)

prompt = ChatPromptTemplate.from_messages([("human", "{content}")])


class StreamableChain:
    """Custom LLM chain for streaming llm generated tokens"""

    def stream(self, input):
        queue = Queue()
        handler = StreamingHandler(queue)

        if not hasattr(self, "__call__") or not callable(self):
            raise NotImplementedError("Chain must be callable")

        def task():
            """Run the chain with the input variables"""
            self(input, callbacks=[handler])

        Thread(target=task).start()

        while True:
            token = queue.get()
            if token is None:
                break
            yield token


class StreamingChain(StreamableChain, LLMChain):
    pass


chain = StreamingChain(llm=chat, prompt=prompt)
for c in chain.stream(input={"content": "tell me a joke"}):
    print(c)
