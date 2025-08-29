"""
Embedding generation module for converting text to vector representations.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Dict, Any, Union, Optional
import os
import numpy as np

class EmbeddingGenerator:
    """Generate embeddings for text using various models."""
    
    def __init__(self, model_name: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the embedding generator with a specific model.
        
        Args:
            model_name: The model to use ('openai' or 'sentence-transformers')
            api_key: Optional API key for OpenAI
        """
        self.model_name = model_name
        
        # Set up the embedding model
        if model_name == "openai":
            # Use OpenAI's embeddings
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            elif "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OpenAI API key is required")
                
            self.model = OpenAIEmbeddings(model="text-embedding-3-small")
            
        elif model_name == "sentence-transformers":
            # Use Hugging Face's sentence transformers (local)
            self.model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single piece of text.
        
        Args:
            text: The text to embed
            
        Returns:
            Vector representation of the text
        """
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self._get_embedding_size()
            
        return self.model.embed_query(text)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of vector representations
        """
        if not texts:
            return []
            
        return self.model.embed_documents(texts)
    
    def generate_document_embeddings(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            Dictionary with text contents, embeddings, and metadata
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.generate_embeddings(texts)
        
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": [doc.metadata for doc in documents]
        }
    
    def _get_embedding_size(self) -> int:
        """Get the dimension size of the embeddings."""
        if self.model_name == "openai":
            return 1536  # OpenAI's text-embedding-3-small dimension
        elif self.model_name == "sentence-transformers":
            return 384   # all-MiniLM-L6-v2 dimension
        else:
            return 1536  # Default