"""
Vector Store Module.

Handles creation and management of ChromaDB vector store with HuggingFace embeddings.
Provides persistent storage and efficient similarity search for PDF document chunks.
"""

import os
from typing import List, Optional, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


# Default embedding model - runs locally (no API required)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default persist directory for ChromaDB
DEFAULT_PERSIST_DIR = "./chroma_db"


class VectorStoreManager:
    """
    Manages ChromaDB vector store for document embeddings.
    
    Handles:
    - Embedding model initialization
    - Vector store creation and persistence
    - Similarity search for retrieval
    - Collection management
    """
    
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = "pdf_documents",
    ):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Name of the HuggingFace embedding model to use
            persist_directory: Directory to persist the vector store
            collection_name: Name of the ChromaDB collection
        """
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._embeddings: Optional[Embeddings] = None
        self._vector_store: Optional[Chroma] = None
    
    @property
    def embeddings(self) -> Embeddings:
        """
        Lazy-load the embedding model locally (no API required).
        
        Returns:
            HuggingFaceEmbeddings instance (runs locally)
        """
        if self._embeddings is None:
            # Use local HuggingFace embeddings (downloads model on first use)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True},  # Better for similarity search
            )
        return self._embeddings
    
    def create_vector_store(
        self,
        documents: List[Document],
        persist: bool = True,
    ) -> Chroma:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to embed and store
            persist: Whether to persist the vector store to disk
            
        Returns:
            Chroma vector store instance
        """
        # Ensure persist directory exists
        if persist and not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)
        
        # Create vector store from documents
        self._vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory if persist else None,
            collection_name=self.collection_name,
        )
        
        return self._vector_store
    
    def load_vector_store(self) -> Optional[Chroma]:
        """
        Load an existing vector store from disk.
        
        Returns:
            Chroma vector store instance, or None if not found
        """
        if not os.path.exists(self.persist_directory):
            return None
        
        try:
            self._vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            return self._vector_store
        except Exception:
            return None
    
    def get_vector_store(self) -> Optional[Chroma]:
        """
        Get the current vector store instance.
        
        Returns:
            Chroma vector store instance, or None if not initialized
        """
        return self._vector_store
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar Document objects
            
        Raises:
            ValueError: If vector store is not initialized
        """
        if self._vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        return self._vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
            
        Raises:
            ValueError: If vector store is not initialized
        """
        if self._vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        return self._vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add new documents to the existing vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
            
        Raises:
            ValueError: If vector store is not initialized
        """
        if self._vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        return self._vector_store.add_documents(documents)
    
    def delete_collection(self) -> None:
        """
        Delete the entire collection from the vector store.
        
        This is useful for clearing all documents when a new PDF is uploaded.
        """
        if self._vector_store is not None:
            self._vector_store.delete_collection()
            self._vector_store = None
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of documents, or 0 if not initialized
        """
        if self._vector_store is None:
            return 0
        
        # ChromaDB doesn't have a direct count method, so we use the collection
        try:
            collection = self._vector_store._collection
            return collection.count()
        except Exception:
            return 0


def create_in_memory_vector_store(
    documents: List[Document],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Chroma:
    """
    Create an in-memory vector store (not persisted to disk).
    
    Useful for temporary storage or testing.
    
    Args:
        documents: List of Document objects to embed and store
        embedding_model: Name of the HuggingFace embedding model to use
        
    Returns:
        Chroma vector store instance (in-memory)
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        # No persist_directory means in-memory
    )
