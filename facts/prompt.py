import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
)
retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
)

query = "What is an interesting fact about the English language?"

result = chain.run(query)

print(result)
