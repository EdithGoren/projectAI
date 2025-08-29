""""
this is a module for extracting text from pdf files
"""
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os

class PDFExtractor:
    """Extract PDF and Metadata"""
    def extract_from_file(self, file_path:str):
        """Extract text from a PDF file
        Args: file path
        Returns: List of documents(Document objects)"""
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        #add basic metadata
        filename = os.path.basename(file_path)
        for doc in documents:
            doc.metadata["filename"] = filename

        return documents
