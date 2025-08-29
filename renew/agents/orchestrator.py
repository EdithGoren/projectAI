"""
Orchestrator for coordinating the multi-agent system for research assistance.
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from .document_agent import DocumentAgent
from .research_agent import ResearchAgent
from .retrieval_agent import RetrievalAgent
from .synthesis_agent import SynthesisAgent
from ..vector_database import VectorStore

class Orchestrator:
    """Coordinates the multi-agent system for research assistance."""
    
    def __init__(self, vector_store: VectorStore, llm_model: str = "gpt-3.5-turbo"):
        """
        Initialize the orchestrator.
        
        Args:
            vector_store: Vector store for document storage and retrieval
            llm_model: Language model to use for agents
        """
        # Initialize the vector store
        self.vector_store = vector_store
        
        # Initialize the agents
        from ..retrieval import Retriever, ContextRetriever, QueryProcessor
        retriever = Retriever(vector_store=vector_store)
        context_retriever = ContextRetriever(retriever=retriever)
        
        self.document_agent = DocumentAgent(vector_store=vector_store, llm_model=llm_model)
        self.research_agent = ResearchAgent(llm_model=llm_model)
        self.retrieval_agent = RetrievalAgent(
            retriever=retriever,
            context_retriever=context_retriever,
            llm_model=llm_model
        )
        self.synthesis_agent = SynthesisAgent(llm_model=llm_model)
        
        # Keep state for workflow
        self.workflow_state = {}
    
    def process_documents(self, input_path: str) -> Dict[str, Any]:
        """
        Process documents for research.
        
        Args:
            input_path: Path to document file or directory
            
        Returns:
            Processing results
        """
        # Determine if the input is a file or directory
        import os
        if os.path.isfile(input_path):
            query = f"Process the document at {input_path}"
        else:
            query = f"Process all documents in the directory at {input_path}"
            
        # Run the document agent
        result = self.document_agent.run(query)
        
        # Update workflow state
        self.workflow_state["documents_processed"] = True
        self.workflow_state["input_path"] = input_path
        
        return result
    
    def research_question(self, question: str) -> Dict[str, Any]:
        """
        Research a question using the multi-agent system.
        
        Args:
            question: Research question to answer
            
        Returns:
            Research results
        """
        # Step 1: Research planning
        print("🔍 Planning research strategy...")
        research_plan = self.research_agent.run(question)
        
        # Step 2: Information retrieval
        print("📚 Retrieving relevant information...")
        retrieval_result = self.retrieval_agent.run(question)
        
        # Get structured context
        context = retrieval_result.get("context", {})
        context_docs = context.get("documents", [])
        
        # Convert to Document objects if needed
        if context_docs and not isinstance(context_docs[0], Document):
            from langchain.schema import Document
            context_docs = [
                Document(page_content=doc["content"], metadata=doc.get("metadata", {}))
                for doc in context_docs
            ]
        
        # Step 3: Response synthesis
        print("✍️ Synthesizing response...")
        if not context_docs:
            print("⚠️ No relevant documents found, using direct retrieval...")
            from ..retrieval import Retriever
            retriever = Retriever(vector_store=self.vector_store)
            context_docs = retriever.retrieve(question, top_k=5)
            
        synthesis_result = self.synthesis_agent.run(question, context_docs)
        
        # Compile final results
        result = {
            "question": question,
            "research_plan": research_plan,
            "relevant_documents": len(context_docs),
            "response": synthesis_result.get("direct_response", {}).get("response", "No response generated"),
            "bibliography": synthesis_result.get("bibliography", [])
        }
        
        return result
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for the orchestrator.
        
        Args:
            query: User query, could be a document processing or research question
            
        Returns:
            Result of processing the query
        """
        # Determine if this is a document processing query or research question
        if any(keyword in query.lower() for keyword in ["process", "document", "pdf", "file", "directory", "index"]):
            # Extract file path from the query
            import re
            path_match = re.search(r'"([^"]+)"|(\S+\.pdf)|(\S+[/\\]\S*)', query)
            
            if path_match:
                file_path = path_match.group(1) or path_match.group(2) or path_match.group(3)
                return self.process_documents(file_path)
            else:
                return {"error": "Could not determine file path from query"}
        else:
            # Treat as a research question
            return self.research_question(query)
