"""
Document agent for processing and indexing documents.
"""

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from ..document_processing import DocumentProcessor
from ..vector_database import EmbeddingGenerator, VectorStore
from typing import Dict, Any
import os

class DocumentAgent:
    """Agent responsible for document processing and indexing."""
    
    def __init__(self, vector_store: VectorStore, llm_model: str = "gpt-3.5-turbo"):
        """
        Initialize the document agent.
        
        Args:
            vector_store: Vector store for document storage
            llm_model: LLM model to use
        """
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = vector_store
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0)
        
        # Define tools available to this agent
        self.tools = [
            Tool(
                name="process_document",
                func=self._process_document,
                description="Process a document file and extract text. Input: file_path"
            ),
            Tool(
                name="process_directory",
                func=self._process_directory,
                description="Process all documents in a directory. Input: directory_path"
            ),
            Tool(
                name="get_document_stats",
                func=self._get_document_stats,
                description="Get statistics about processed documents. No input required."
            )
        ]
        
        # Create agent prompt
        self.agent_prompt = PromptTemplate.from_template(
            """You are a Document Processing Agent responsible for handling academic documents.
            Your job is to process PDF files, extract their content, and index them for later retrieval.
            
            Use the following tools to accomplish your tasks:
            {tools}
            
            Follow this process:
            1. Understand what documents need to be processed
            2. Process them using the appropriate tool
            3. Verify the documents were indexed correctly
            
            Use the format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Question: {input}
            {agent_scratchpad}
            """
        )
        
        # Create the agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.agent_prompt
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True  # Add this parameter to handle parsing errors
        )
    
    def _process_document(self, file_path: str) -> str:
        """Process a single document file."""
        try:
            # Fix potential path formatting issues from agent input
            # Remove any 'file_path = ' prefix that might come from agent
            if "file_path =" in file_path:
                file_path = file_path.split("file_path =")[1].strip().strip('"')
            
            # Normalize path separators for Windows
            file_path = os.path.normpath(file_path)
            
            # Process the document
            chunks = self.document_processor.process_document(file_path)
            
            # Add to vector store
            ids = self.vector_store.add_documents(chunks)
            
            return f"Successfully processed document: {file_path}. Created {len(chunks)} chunks and added to vector store."
        except Exception as e:
            return f"Error processing document: {str(e)}"
    
    def _process_directory(self, directory_path: str) -> str:
        """Process all documents in a directory."""
        try:
            # Fix potential path formatting issues from agent input
            # Remove any 'directory_path = ' prefix that might come from agent
            if "directory_path =" in directory_path:
                directory_path = directory_path.split("directory_path =")[1].strip().strip('"')
            
            # Normalize path separators for Windows
            directory_path = os.path.normpath(directory_path)
            
            # Process the directory
            chunks = self.document_processor.process_directory(directory_path)
            
            # Add to vector store
            ids = self.vector_store.add_documents(chunks)
            
            return f"Successfully processed directory: {directory_path}. Processed {len(chunks)} chunks from documents."
        except Exception as e:
            return f"Error processing directory: {str(e)}"
    
    def _get_document_stats(self) -> str:
        """Get statistics about processed documents."""
        try:
            # This would need to be implemented in VectorStore
            # For now, return a placeholder
            return "Document statistics not yet implemented."
        except Exception as e:
            return f"Error getting statistics: {str(e)}"
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the document agent with a query."""
        return self.agent_executor.invoke({"input": query})
