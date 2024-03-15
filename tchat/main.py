import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=openai_api_key)

prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[HumanMessagePromptTemplate.from_template("{content}")],
)

chain = LLMChain(llm=chat, prompt=prompt)

while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])
