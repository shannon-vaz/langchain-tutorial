import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)
db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="emb")

print("Embeddings created.")
