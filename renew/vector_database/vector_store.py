"""
Vector storage and retrieval using Pinecone.
"""

from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from .embedding import EmbeddingGenerator
from typing import List, Dict, Any, Optional
import os
import uuid

class VectorStore:
    """Store and retrieve vectors using Pinecone."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        index_name: str = "research-assistant",
        namespace: str = "default",
        embedding_model: str = "openai"
    ):
        """
        Initialize the vector store.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index to use
            namespace: Namespace within the index
            embedding_model: Model to use for embeddings
        """
        # Set up Pinecone client
        if api_key:
            self.api_key = api_key
        elif "PINECONE_API_KEY" in os.environ:
            self.api_key = os.environ["PINECONE_API_KEY"]
        else:
            raise ValueError("Pinecone API key is required")
        
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize the embedding generator
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        
        # Initialize Pinecone
        self.pinecone_client = Pinecone(api_key=self.api_key)
        
        # Check if index exists, if not create it
        self._create_index_if_not_exists()
        
        # Set up LangChain's PineconeVectorStore
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            namespace=self.namespace,
            embedding=self.embedding_generator.model
        )
    
    def _create_index_if_not_exists(self):
        """Create Pinecone index if it doesn't already exist."""
        # List existing indexes
        existing_indexes = [
            index_info["name"] for index_info in self.pinecone_client.list_indexes()
        ]
        
        # Create new index if it doesn't exist
        if self.index_name not in existing_indexes:
            # Get embedding dimension from the model
            embedding_dimension = self.embedding_generator._get_embedding_size()
            
            # Create the index
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of IDs for the added documents
        """
        # Generate unique IDs for documents
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Add documents to vector store
        self.vector_store.add_documents(documents=documents, ids=ids)
        
        return ids
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Search for similar documents given a query string.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(query, k=top_k)
    
    def similarity_search_with_score(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        return self.vector_store.similarity_search_with_score(query, k=top_k)
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store by ID.
        
        Args:
            ids: List of document IDs to delete
        """
        self.vector_store.delete(ids)
        
    def clear_namespace(self) -> None:
        """Delete all documents in the current namespace."""
        try:
            pinecone_index = self.pinecone_client.Index(self.index_name)
            pinecone_index.delete(delete_all=True, namespace=self.namespace)
            print(f"Cleared namespace '{self.namespace}'")
        except Exception as e:
        # Handle case where namespace doesn't exist yet
            if "Namespace not found" in str(e):
                print(f"Namespace '{self.namespace}' does not exist yet - nothing to clear")
            else:
            # Re-raise other unexpected errors
                raise