from typing import List
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever
from langchain_core.documents import Document


class RedundantFilterRetriever(BaseRetriever):
    """Custom Retriever to perform retrieval while filtering redundant
    documents."""
    embeddings: Embeddings
    chroma: Chroma

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # Calculate embeddings for the query
        emb = self.embeddings.embed_query(query)

        # Filter out redundant documents using max marginal relevance
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb, lambda_mult=0.8
        )
