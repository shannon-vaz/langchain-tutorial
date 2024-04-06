from queue import Queue
from threading import Thread
from flask import current_app
from app.chat.callbacks.stream import StreamingHandler


class StreamableChain:
    """Mixin for making LLM chain for streaming llm generated tokens"""

    def stream(self, input):
        queue = Queue()
        handler = StreamingHandler(queue)

        if not hasattr(self, "__call__") or not callable(self):
            raise NotImplementedError("Chain must be callable")

        def task(app_context):
            """Run the chain with the input variables"""
            app_context.push()
            self(input, callbacks=[handler])

        Thread(target=task, args=[current_app.app_context()]).start()

        while True:
            token = queue.get()
            if token is None:
                break
            yield token
