"""
Citation management module for tracking and formatting citations.
"""

from langchain.schema import Document
from typing import List, Dict, Any, Optional
import re
import hashlib

class CitationManager:
    """Manage and format citations from documents."""
    
    def __init__(self):
        """Initialize the citation manager."""
        # Dictionary to store citation information
        self.citations = {}
        
        # Citation formats
        self.formats = {
            "inline": "[{id}]",
            "academic": "[{author}, {year}]",
            "numbered": "[{num}]",
            "footnote": "^{num}"
        }
    
    def add_citations(self, documents: List[Document]) -> Dict[str, str]:
        """
        Add documents to citation manager and return citation keys.
        
        Args:
            documents: List of documents to cite
            
        Returns:
            Dictionary mapping document indices to citation keys
        """
        citation_keys = {}
        
        for i, doc in enumerate(documents):
            # Generate a unique citation key
            citation_key = self._generate_citation_key(doc)
            
            # Store citation information
            if citation_key not in self.citations:
                self.citations[citation_key] = {
                    "document": doc,
                    "index": i,
                    "metadata": doc.metadata,
                    "content_hash": self._hash_content(doc.page_content)
                }
                
            citation_keys[i] = citation_key
            
        return citation_keys
    
    def _generate_citation_key(self, doc: Document) -> str:
        """
        Generate a unique citation key for a document.
        
        Args:
            doc: Document to generate key for
            
        Returns:
            Unique citation key
        """
        # Try to use author and year if available
        author = doc.metadata.get("author", "")
        year = doc.metadata.get("year", "")
        source = doc.metadata.get("source", "")
        
        if author and year:
            # Use first author's last name and year
            first_author = author.split(",")[0].strip()
            last_name = first_author.split()[-1]
            return f"{last_name}{year}"
        elif source:
            # Use source filename without extension
            filename = source.split("/")[-1].split(".")[0]
            return filename
        else:
            # Generate a hash-based key from content
            content_hash = self._hash_content(doc.page_content)
            return f"doc-{content_hash[:6]}"
    
    def _hash_content(self, content: str) -> str:
        """
        Generate a hash of content for identification.
        
        Args:
            content: Text content to hash
            
        Returns:
            Hash string
        """
        return hashlib.md5(content.encode()).hexdigest()
    
    def format_citation(self, citation_key: str, format_type: str = "inline") -> str:
        """
        Format a citation according to the specified format.
        
        Args:
            citation_key: Citation key to format
            format_type: Citation format to use
            
        Returns:
            Formatted citation string
        """
        if citation_key not in self.citations:
            return f"[Unknown citation]"
            
        citation = self.citations[citation_key]
        
        # Get format template
        template = self.formats.get(format_type, self.formats["inline"])
        
        # Extract citation information
        metadata = citation["metadata"]
        author = metadata.get("author", "Unknown")
        year = metadata.get("year", "")
        source = metadata.get("source", "Unknown")
        index = citation["index"]
        
        # Format based on available information and format type
        return template.format(
            id=citation_key,
            author=author.split(",")[0].strip() if "," in author else author,
            year=year,
            num=index + 1,  # 1-based indexing for readers
            source=source
        )
    
    def get_bibliography(self, format_type: str = "academic") -> List[Dict[str, str]]:
        """
        Generate a bibliography of all citations.
        
        Args:
            format_type: Citation format style
            
        Returns:
            List of bibliography entries
        """
        bibliography = []
        
        for key, citation in sorted(self.citations.items(), key=lambda x: x[1]["index"]):
            metadata = citation["metadata"]
            doc = citation["document"]
            
            # Extract citation information
            author = metadata.get("author", "Unknown")
            year = metadata.get("year", "")
            title = metadata.get("title", "")
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "")
            
            # Create bibliography entry
            entry = {
                "key": key,
                "citation": self.format_citation(key, format_type),
                "author": author,
                "year": year,
                "title": title,
                "source": source,
                "page": page,
                "text_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            }
            
            bibliography.append(entry)
            
        return bibliography
    
    def find_citations_in_text(self, text: str) -> List[str]:
        """
        Find all citation keys in text.
        
        Args:
            text: Text to search for citations
            
        Returns:
            List of citation keys found in text
        """
        # Look for common citation patterns
        patterns = [
            r'\[([A-Za-z]+\d{4})\]',  # [Smith2020]
            r'\[([A-Za-z\-]+)\]',      # [Smith]
            r'\[doc-([a-f0-9]{6})\]',  # [doc-1a2b3c]
            r'\[Document\s+(\d+)\]'    # [Document 1]
        ]
        
        found_citations = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            found_citations.extend(matches)
        
        # Filter to only include keys in our citation database
        return [citation for citation in found_citations 
                if citation in self.citations or
                (citation.isdigit() and int(citation)-1 < len(self.citations))]
