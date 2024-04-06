import os
from langchain.chat_models import ChatOpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")


def build_llm(chat_args):
    return ChatOpenAI(openai_api_key=openai_api_key, streaming=chat_args.streaming)
