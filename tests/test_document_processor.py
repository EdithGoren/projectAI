"""
Test script for the document processor.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fix the import - can't use hyphens in import statements
sys.path.append('..')
# Use direct import
from research_assistant.document_processing.document_processor import DocumentProcessor

def main():
    # Initialize the document processor
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # Define data directory
    data_dir = os.path.join("data", "raw")
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if there are any PDF files
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {data_dir}. Please add some PDF files.")
        return
    
    # Process the first PDF file
    first_pdf = os.path.join(data_dir, pdf_files[0])
    print(f"Processing {first_pdf}...")
    
    # Process the document
    processed_docs = processor.process_document(first_pdf)
    
    # Display results
    print(f"Document processed into {len(processed_docs)} chunks")
    print("\nSample chunk:")
    if processed_docs:
        print(f"Chunk content: {processed_docs[0].page_content[:200]}...")
        print(f"Chunk metadata: {dict(processed_docs[0].metadata)}")
    
if __name__ == "__main__":
    main()