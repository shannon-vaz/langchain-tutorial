import os
import pinecone
from langchain.vectorstores import Pinecone
from app.chat.embeddings.openai import embeddings

pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
pinecone_env_name = os.getenv("PINECONE_ENV_NAME", "")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_env_name,
)

vector_store = Pinecone.from_existing_index(pinecone_index_name, embeddings)


def build_retriever(chat_args, k):
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id}, "k": k}
    return vector_store.as_retriever(search_kwargs=search_kwargs)
