"""
Response generation module for creating answers from retrieved context.
"""

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import re
import os

class ResponseGenerator:
    """Generate coherent responses from retrieved context."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the response generator.
        
        Args:
            model_name: The LLM model to use for response generation
        """
        # Get API key if available
        api_key = os.environ.get("OPENAI_API_KEY")
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.2,  # Slightly higher temperature for more natural responses
            api_key=api_key
        )
        
        # Flag for test mode
        self._is_test_mode = api_key is None
        
        # Create response generation prompt template
        self.response_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            You are an academic research assistant helping with a query.
            Use the following pieces of context to answer the user query.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            Always include citations from the sources in your answer using [Document X] notation.
            
            Query: {query}
            
            Context:
            {context}
            
            Answer:
            """
        )
    
    def generate_response(self, query: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved documents.
        
        Args:
            query: User query
            context_docs: List of retrieved documents
            
        Returns:
            Dictionary with response and metadata
        """
        # Prepare context from documents
        context_text = self._prepare_context(context_docs)
        
        # Generate response
        response_text = self._generate_text(query, context_text)
        
        # Extract citations
        citations = self._extract_citations(response_text, context_docs)
        
        return {
            "query": query,
            "response": response_text,
            "citations": citations,
            "sources": list(set(doc.metadata.get("source", "Unknown") for doc in context_docs))
        }
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Prepare context from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Format each document with a label
            doc_text = f"[Document {i+1}]"
            
            # Add source information if available
            if "source" in doc.metadata:
                doc_text += f" Source: {doc.metadata['source']}"
                
            # Add page information if available
            if "page" in doc.metadata:
                doc_text += f" (Page {doc.metadata['page']})"
                
            # Add the actual content
            doc_text += f"\n{doc.page_content}\n"
            
            context_parts.append(doc_text)
            
        return "\n\n".join(context_parts)
    
    def _generate_text(self, query: str, context: str) -> str:
        """
        Generate response text using the LLM.
        
        Args:
            query: User query
            context: Context from retrieved documents
            
        Returns:
            Generated response text
        """
        # Prepare inputs
        inputs = {
            "query": query,
            "context": context
        }
        
        # Run the model - handle test mode if API key not available
        if self._is_test_mode:
            return "This is a test response generated without an API call. In this response, I've found information from [Document 1] and [Document 2]."
        else:
            # Run the model
            result = self.llm.invoke(self.response_template.format(**inputs))
            # Extract the text content
            return result.content
    
    def _extract_citations(self, text: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Extract citations from response text.
        
        Args:
            text: Response text
            documents: Original documents
            
        Returns:
            Dictionary of citation information
        """
        # Find all citation references like [Document X]
        citation_pattern = r'\[Document\s+(\d+)\]'
        matches = re.findall(citation_pattern, text)
        
        # Convert to integers and deduplicate
        doc_indices = sorted(set(int(idx) for idx in matches))
        
        # Build citation information
        citations = {}
        for idx in doc_indices:
            # Check if the index is valid
            if idx <= len(documents) and idx > 0:
                doc = documents[idx-1]  # Convert to 0-based index
                doc_id = f"doc{idx}"
                
                citations[doc_id] = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", None)
                }
                
        return citations
