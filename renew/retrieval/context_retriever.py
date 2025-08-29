"""
Context retriever module for preparing retrieved context for response generation.
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from .retriever import Retriever

class ContextRetriever:
    """Prepares retrieved context for response generation."""
    
    def __init__(self, retriever: Retriever, max_tokens: int = 3000):
        """
        Initialize the context retriever.
        
        Args:
            retriever: The retriever to use for document retrieval
            max_tokens: Maximum number of tokens to include in context
        """
        self.retriever = retriever
        self.max_tokens = max_tokens
    
    def get_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Get context for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
        
        Returns:
            Dictionary containing context and metadata
        """
        # Retrieve documents
        documents = self.retriever.retrieve(query, top_k=top_k)
        
        # Format context
        formatted_context = self._format_context(documents)
        
        # Build result
        result = {
            "query": query,
            "documents": documents,
            "context": formatted_context,
            "document_count": len(documents)
        }
        
        return result
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents: List of retrieved documents
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Format document with metadata
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "")
            page_info = f", page {page}" if page else ""
            
            # Add document to context
            context_parts.append(
                f"[Document {i+1}] From {source}{page_info}:\n{doc.page_content}\n"
            )
        
        # Join all parts
        context = "\n\n".join(context_parts)
        
        # Truncate if needed (approximate token count)
        approx_tokens = len(context) / 4  # rough approximation
        if approx_tokens > self.max_tokens:
            ratio = self.max_tokens / approx_tokens
            context = context[:int(len(context) * ratio)]
            context += "\n\n[Context truncated due to length limits]"
        
        return context