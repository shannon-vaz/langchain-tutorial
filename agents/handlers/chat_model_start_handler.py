from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen


def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))


class ChatModelStartHandler(BaseCallbackHandler):

    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n\n\n\n====== Sending Messages ======\n\n")

        # Assume messages are not batched while calling LLM
        for message in messages[0]:
            if message.type == "system":  # System message
                boxen_print(message.content, title=message.type, color="yellow")
            elif message.type == "human":  # Human message
                boxen_print(message.content, title=message.type, color="green")
            elif (
                message.type == "ai" and "function_call" in message.additional_kwargs
            ):  # AI message with function call request
                call = message.additional_kwargs["function_call"]
                boxen_print(
                    f"Running tool {call['name']} with args {call['arguments']}",
                    title=message.type,
                    color="cyan",
                )
            elif message.type == "ai":  # AI message
                boxen_print(message.content, title=message.type, color="blue")
            elif message.type == "function":  # Function result
                boxen_print(message.content, title=message.type, color="purple")
            else:  # Unknown
                boxen_print(message.content, title=message.type)
