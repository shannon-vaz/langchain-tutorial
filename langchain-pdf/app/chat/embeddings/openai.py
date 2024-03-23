import os
from langchain.embeddings import OpenAIEmbeddings

openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
