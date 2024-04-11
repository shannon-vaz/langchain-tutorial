from app.chat.tracing.langfuse import langfuse
from langfuse.model import CreateTrace


class TraceableChain:
    """
    Mixin to override the call method to pass the tracing callback handler to
    chain at call time.
    This ensures the trace callback handler is called for all the components and
    subcomponents of the chain.
    """

    def __call__(self, *args, **kwargs):
        trace = langfuse.trace(
            CreateTrace(
                id=self.metadata["conversation_id"],
                metadata=self.metadata,
            )
        )
        callbacks = kwargs.get("callbacks", [])
        callbacks.append(trace.getNewHandler())
        kwargs["callbacks"] = callbacks
        return super().__call__(*args, **kwargs)  # type: ignore
