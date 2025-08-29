"""
Document processing module using LangChain for PDF extraction, cleaning, and chunking.
"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import os
import re

class DocumentProcessor:
    """Process documents using LangChain."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configure text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Replace multiple spaces with single space
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Remove form feed characters
        cleaned = cleaned.replace('\f', ' ')
        
        # Fix line breaks in sentences
        cleaned = re.sub(r'(\w+)\s*\n\s*(\w+)', r'\1 \2', cleaned)
        
        return cleaned.strip()
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Process a single document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of processed document chunks
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Load the document with PyMuPDF
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # Clean text in documents
        for doc in documents:
            doc.page_content = self._clean_text(doc.page_content)
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Enhance metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = str(i)
            chunk.metadata["filename"] = os.path.basename(file_path)
            chunk.metadata["file_path"] = file_path
            
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of processed document chunks
        """
        # Check if directory exists
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_documents = []
        
        # Process each PDF file in the directory
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                processed_docs = self.process_document(file_path)
                all_documents.extend(processed_docs)
        
        return all_documents
    
    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary of document metadata
        """
        # Load the document
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            return {}
        
        # Extract basic metadata
        metadata = documents[0].metadata.copy()
        
        # Enhance with additional metadata
        metadata.update({
            "filename": os.path.basename(file_path),
            "file_path": file_path,
            "total_pages": len(documents),
        })
        
        return metadata