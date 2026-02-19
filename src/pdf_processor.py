"""
PDF Processing Module.

Handles PDF loading, text extraction, and chunking for vector storage.
Uses PyPDF for parsing and LangChain's text splitters for chunking.
"""

from io import BytesIO
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFProcessor:
    """
    Processes PDF files for RAG applications.
    
    Handles:
    - PDF loading and text extraction
    - Text chunking with configurable parameters
    - Metadata extraction (page numbers, source)
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Maximum size of each text chunk (default: 1000 characters)
            chunk_overlap: Number of characters to overlap between chunks (default: 200)
            separators: Custom separators for text splitting (default: ["\n\n", "\n", " ", ""])
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        # Initialize the text splitter with optimized settings for PDF content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )
    
    def load_pdf(self, pdf_file: BytesIO, filename: str = "uploaded.pdf") -> Tuple[List[Document], int]:
        """
        Load and extract text from a PDF file.
        
        Args:
            pdf_file: BytesIO object containing the PDF data
            filename: Name of the file for metadata
            
        Returns:
            Tuple of (list of Document objects, total page count)
            
        Raises:
            ValueError: If PDF cannot be parsed or is empty
        """
        try:
            # Save BytesIO to a temporary file for PyPDFLoader
            # PyPDFLoader requires a file path, so we use a temporary approach
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Load PDF using PyPDFLoader
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                
                if not pages:
                    raise ValueError("PDF appears to be empty or cannot be parsed")
                
                # Add source filename to metadata
                for page in pages:
                    page.metadata["source"] = filename
                
                return pages, len(pages)
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            raise ValueError(f"Failed to load PDF: {str(e)}")
    
    def chunk_documents(
        self, 
        documents: List[Document],
        add_start_index: bool = True
    ) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of Document objects to chunk
            add_start_index: Whether to add start index to metadata
            
        Returns:
            List of chunked Document objects with preserved metadata
        """
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Preserve and enhance metadata
        for i, chunk in enumerate(chunks):
            # Ensure metadata exists
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            
            # Add chunk index for reference
            chunk.metadata['chunk_index'] = i
            
            # Add start index if requested (helps with citation)
            if add_start_index and hasattr(chunk, 'metadata'):
                chunk.metadata['start_index'] = i * (self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    def process_pdf(self, pdf_file: BytesIO, filename: str = "uploaded.pdf") -> Tuple[List[Document], dict]:
        """
        Complete PDF processing pipeline: load, extract, and chunk.
        
        Args:
            pdf_file: BytesIO object containing the PDF data
            filename: Name of the file for metadata
            
        Returns:
            Tuple of (chunked documents, processing metadata)
            
        Raises:
            ValueError: If processing fails
        """
        # Load PDF
        pages, page_count = self.load_pdf(pdf_file, filename)
        
        # Chunk documents
        chunks = self.chunk_documents(pages)
        
        # Prepare metadata
        metadata = {
            "filename": filename,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        
        return chunks, metadata


def get_pdf_stats(documents: List[Document]) -> dict:
    """
    Get statistics about processed PDF documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Dictionary with statistics (total chars, avg chunk size, etc.)
    """
    if not documents:
        return {"total_chunks": 0, "total_chars": 0, "avg_chunk_size": 0}
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    
    return {
        "total_chunks": len(documents),
        "total_chars": total_chars,
        "avg_chunk_size": total_chars // len(documents),
        "min_chunk_size": min(len(doc.page_content) for doc in documents),
        "max_chunk_size": max(len(doc.page_content) for doc in documents),
    }
