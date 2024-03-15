from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(
    openai_api_key=openai_api_key
)
result = llm("Write a very very short poem")
print(result)