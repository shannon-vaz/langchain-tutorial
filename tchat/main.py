import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=openai_api_key)
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory(
        file_path="chat_history.json",
    ),
    memory_key="messages",
    return_messages=True,
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory)

while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])
