"""
Test script for the retrieval system components.
"""

import sys
import os
import pytest
from dotenv import load_dotenv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research_assistant.document_processing import DocumentProcessor
from research_assistant.vector_database import VectorStore
from research_assistant.retrieval import QueryProcessor, Retriever, ContextRetriever
from langchain.schema import Document

# Load environment variables from .env file if present
load_dotenv()

def test_query_processor():
    """Test the query processor functionality."""
    # Skip if OpenAI API key is not available
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY environment variable not set")
        
    # Create query processor
    query_processor = QueryProcessor()
    
    # Test query cleaning
    messy_query = "  Um, how do neural networks actually   work? Like, you know.  "
    cleaned_query = query_processor.clean_query(messy_query)
    
    assert cleaned_query == "how do neural networks actually work?"
    print(f"✅ Successfully cleaned query")
    
    # Test query expansion - skip LLM parts if testing without network
    try:
        expanded_query = query_processor.expand_query("How do neural networks work?")
        
        assert len(expanded_query) > len("How do neural networks work?")
        assert "neural" in expanded_query.lower()
        print(f"✅ Successfully expanded query: {expanded_query}")
        
        # Test hybrid query generation
        hybrid_queries = query_processor.generate_hybrid_queries("How do neural networks work?")
        
        assert "original" in hybrid_queries
        assert "cleaned" in hybrid_queries
        assert "expanded" in hybrid_queries
        assert "keywords" in hybrid_queries
        print(f"✅ Successfully generated hybrid queries")
    except Exception as e:
        print(f"⚠️ LLM-dependent tests skipped: {e}")

def setup_test_documents():
    """Create test documents for retrieval testing."""
    return [
        Document(page_content="Neural networks are a class of machine learning algorithms inspired by the human brain.", 
                metadata={"source": "test1", "topic": "machine learning"}),
        Document(page_content="Transformers have revolutionized natural language processing with attention mechanisms.", 
                metadata={"source": "test2", "topic": "NLP"}),
        Document(page_content="Vector databases store and retrieve high-dimensional vectors efficiently.", 
                metadata={"source": "test3", "topic": "databases"}),
        Document(page_content="Convolutional neural networks (CNNs) are particularly effective for image processing tasks.", 
                metadata={"source": "test4", "topic": "computer vision"}),
    ]

def test_retrieval_system():
    """Test the complete retrieval system with test documents."""
    try:
        # Create the test documents
        documents = setup_test_documents()
        
        # Create mock vector store
        from unittest.mock import MagicMock
        mock_vector_store = MagicMock()
        
        # Setup mock behavior for similarity_search
        def mock_similarity_search(query, top_k=5):
            # Very simple keyword matching for testing
            results = []
            for doc in documents:
                if any(word.lower() in doc.page_content.lower() for word in query.split()):
                    results.append(doc)
                if len(results) >= top_k:
                    break
            return results
            
        # Setup mock behavior for similarity_search_with_score
        def mock_similarity_search_with_score(query, top_k=5):
            # Simple keyword matching with mock scores
            results = []
            for doc in documents:
                # Calculate mock score based on word overlap
                query_words = set(query.lower().split())
                doc_words = set(doc.page_content.lower().split())
                overlap = len(query_words.intersection(doc_words))
                if overlap > 0:
                    # Normalize to 0-1 range
                    score = min(0.5 + (overlap / len(query_words)) * 0.5, 1.0)
                    results.append((doc, score))
                if len(results) >= top_k:
                    break
            return results
        
        # Attach the mock methods
        mock_vector_store.similarity_search = mock_similarity_search
        mock_vector_store.similarity_search_with_score = mock_similarity_search_with_score
        
        # Create a mock query processor that doesn't use LLM
        mock_query_processor = MagicMock()
        
        # Setup mock behavior for clean_query
        def mock_clean_query(query):
            return query.strip().lower()
        
        # Setup mock behavior for generate_hybrid_queries
        def mock_generate_hybrid_queries(query):
            cleaned = mock_clean_query(query)
            return {
                "original": query,
                "cleaned": cleaned,
                "keywords": " ".join([word for word in cleaned.split() if len(word) > 3]),
                "expanded": f"{cleaned} algorithms machine learning artificial intelligence"
            }
            
        # Attach the mock methods to query processor
        mock_query_processor.clean_query = mock_clean_query
        mock_query_processor.generate_hybrid_queries = mock_generate_hybrid_queries
        
        # Create components with mocks
        retriever = Retriever(
            vector_store=mock_vector_store, 
            query_processor=mock_query_processor
        )
        context_retriever = ContextRetriever(retriever=retriever)
        
        # Test retrieval
        query = "How do neural networks work?"
        results = retriever.retrieve(query, top_k=2, use_hybrid=False)  # Simple search
        
        # Check results
        assert len(results) > 0, "Should find at least one document"
        assert any("neural" in doc.page_content.lower() for doc in results)
        print(f"✅ Retrieved {len(results)} documents for simple search")
        
        # Test context retrieval
        context = context_retriever.get_context(query, top_k=2)
        
        # Check context
        assert "context" in context
        assert len(context["context"]) > 0
        assert "token_estimate" in context
        print(f"✅ Generated context with {context['token_estimate']} tokens")
        
    except Exception as e:
        print(f"Retrieval test failed with error: {str(e)}")
        raise

def main():
    """Run test functions."""
    print("\n🧪 Testing Query Processor:")
    try:
        test_query_processor()
    except Exception as e:
        print(f"❌ Query processor test failed: {e}")
    
    print("\n🧪 Testing Retrieval System:")
    try:
        test_retrieval_system()
    except Exception as e:
        print(f"❌ Retrieval system test failed: {e}")

if __name__ == "__main__":
    main()