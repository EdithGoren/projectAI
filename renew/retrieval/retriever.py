"""
Advanced retrieval system for semantic search.
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from .query_processor import QueryProcessor
from ..vector_database.vector_store import VectorStore
import os

class Retriever:
    """Advanced retrieval system for finding relevant documents."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        query_processor: Optional[QueryProcessor] = None,
        relevance_threshold: float = 0.7
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store for document retrieval
            query_processor: Query processor for query optimization (optional)
            relevance_threshold: Minimum similarity score for inclusion (0-1)
        """
        self.vector_store = vector_store
        self.query_processor = query_processor or QueryProcessor()
        self.relevance_threshold = relevance_threshold
        
        # Initialize LLM for reranking if API key is available
        if "OPENAI_API_KEY" in os.environ:
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", 
                temperature=0,
                api_key=os.environ.get("OPENAI_API_KEY")
            )
            
            # Create reranking prompt
            self.reranking_template = PromptTemplate(
                input_variables=["query", "results"],
                template="""
                You are an academic research assistant. Rate the relevance of each document to the query on a scale from 0 to 10.
                
                Query: {query}
                
                Documents:
                {results}
                
                Return only a Python list of scores, one score per document. Example: [7, 4, 9]
                """
            )
            
            self.reranking_chain = LLMChain(llm=self.llm, prompt=self.reranking_template)
        else:
            self.llm = None
            self.reranking_chain = None
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        use_hybrid: bool = True,
        use_reranking: bool = False
    ) -> List[Document]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            use_hybrid: Whether to use hybrid search
            use_reranking: Whether to use LLM reranking
            
        Returns:
            List of relevant documents
        """
        if use_hybrid:
            # Generate multiple query variants
            queries = self.query_processor.generate_hybrid_queries(query)
            
            # Run multiple searches
            all_results = []
            for query_type, query_text in queries.items():
                try:
                    results = self.vector_store.similarity_search_with_score(query_text, top_k=top_k)
                    for doc, score in results:
                        # Add query type to metadata
                        doc.metadata["query_type"] = query_type
                        doc.metadata["similarity_score"] = float(score)
                        all_results.append((doc, score))
                except Exception as e:
                    print(f"Search failed for query type '{query_type}': {e}")
            
            # Filter by threshold and deduplicate
            filtered_results = []
            seen_contents = set()
            
            for doc, score in all_results:
                if score >= self.relevance_threshold and doc.page_content not in seen_contents:
                    filtered_results.append((doc, score))
                    seen_contents.add(doc.page_content)
            
            # Sort by score
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            
            # Take top results
            results = filtered_results[:top_k]
            documents = [doc for doc, _ in results]
            
            # Apply reranking if requested and available
            if use_reranking and self.reranking_chain and documents:
                documents = self._rerank_documents(query, documents)
            
            return documents
            
        else:
            # Simple search
            try:
                return self.vector_store.similarity_search(query, top_k=top_k)
            except Exception as e:
                print(f"Simple search failed: {e}")
                return []
    
    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using an LLM.
        
        Args:
            query: User query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        if not self.reranking_chain:
            return documents
            
        # Prepare document content for the prompt
        doc_texts = [f"Document {i+1}: {doc.page_content[:300]}..." for i, doc in enumerate(documents)]
        doc_text = "\n\n".join(doc_texts)
        
        # Run reranking
        try:
            result = self.reranking_chain.run(query=query, results=doc_text)
            # Parse scores from the result (assuming format like [7, 4, 9])
            scores = eval(result)
            
            # Check if we got valid scores
            if len(scores) == len(documents) and all(isinstance(s, (int, float)) for s in scores):
                # Create (document, score) pairs
                doc_scores = list(zip(documents, scores))
                # Sort by score in descending order
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                # Extract just the documents
                return [doc for doc, _ in doc_scores]
        except Exception as e:
            print(f"Reranking failed: {e}")
        
        # Fall back to original order if reranking fails
        return documents
    
    def retrieve_with_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve documents with additional context information.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with documents and context
        """
        # Process the query
        cleaned_query = self.query_processor.clean_query(query)
        
        # Get relevant documents
        documents = self.retrieve(cleaned_query, top_k=top_k)
        
        # Extract metadata summaries
        sources = []
        topics = set()
        for doc in documents:
            if "source" in doc.metadata:
                sources.append(doc.metadata["source"])
            if "topic" in doc.metadata:
                topics.add(doc.metadata["topic"])
        
        # Return documents and context
        return {
            "original_query": query,
            "processed_query": cleaned_query,
            "documents": documents,
            "sources": list(set(sources)),
            "topics": list(topics),
            "count": len(documents)
        }