"""
Agent Tools Module.

Defines the two main tools for the PDF QA Agent:
1. PDF Retrieval Tool - For document-specific questions
2. General LLM Tool - For general knowledge questions

These tools are used by the LangChain agent for intelligent routing.
"""

from typing import Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from src.vector_store import VectorStoreManager


class PDFRetrievalInput(BaseModel):
    """Input schema for PDF Retrieval Tool."""
    
    query: str = Field(
        default="",
        description="The question or query to search for in the uploaded PDF document"
    )


class GeneralLLMInput(BaseModel):
    """Input schema for General LLM Tool."""
    
    query: str = Field(
        default="",
        description="The general knowledge question to answer"
    )


class PDFRetrievalTool(BaseTool):
    """
    Tool for retrieving information from uploaded PDF documents.
    
    This tool performs semantic search over the PDF content to find
    relevant passages and returns context-aware answers.
    
    Use this tool when:
    - Questions reference specific document content
    - User asks about information in the uploaded PDF
    - Query requires document-specific context
    """
    
    name: str = "pdf_retrieval"
    description: str = (
        "Search and retrieve information from the uploaded PDF document. "
        "Use this tool when the user's question is about content in the uploaded PDF, "
        "requires specific document context, or references figures, tables, or sections "
        "from the document. Input should be a clear question or search query."
    )
    args_schema: Type[BaseModel] = PDFRetrievalInput
    
    # Vector store manager instance
    vector_store_manager: Optional[VectorStoreManager] = None
    # Number of relevant chunks to retrieve
    k: int = 4
    
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        k: int = 4,
        **kwargs
    ):
        """
        Initialize the PDF Retrieval Tool.
        
        Args:
            vector_store_manager: Manager for the vector store
            k: Number of relevant chunks to retrieve (default: 4)
        """
        super().__init__(**kwargs)
        self.vector_store_manager = vector_store_manager
        self.k = k
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Execute the PDF retrieval.
        
        Args:
            query: The search query
            run_manager: Callback manager (unused but required by BaseTool)
            
        Returns:
            Retrieved context as a formatted string
        """
        if self.vector_store_manager is None:
            return "Error: No PDF has been uploaded. Please upload a PDF first."
        
        vector_store = self.vector_store_manager.get_vector_store()
        if vector_store is None:
            return "Error: No PDF has been processed. Please upload a PDF first."
        
        try:
            # Perform similarity search with scores
            results = self.vector_store_manager.similarity_search_with_score(
                query=query,
                k=self.k,
            )
            
            if not results:
                return "No relevant information found in the uploaded document."
            
            # Format the retrieved context
            context_parts = []
            for i, (doc, score) in enumerate(results, 1):
                # Get page number if available
                page_num = doc.metadata.get('page', 'Unknown')
                source = doc.metadata.get('source', 'Unknown')
                
                context_parts.append(
                    f"[Excerpt {i} (Page {page_num}, Relevance: {score:.3f})]\n"
                    f"{doc.page_content}\n"
                )
            
            context = "\n".join(context_parts)
            
            return (
                f"Found {len(results)} relevant passages from the document:\n\n"
                f"{context}\n\n"
                f"Use this context to answer the user's question."
            )
            
        except Exception as e:
            return f"Error searching document: {str(e)}"
    
    def update_vector_store(self, vector_store_manager: VectorStoreManager) -> None:
        """
        Update the vector store manager reference.
        
        Args:
            vector_store_manager: New vector store manager instance
        """
        self.vector_store_manager = vector_store_manager


class GeneralLLMTool(BaseTool):
    """
    Tool for answering general knowledge questions.
    
    This tool provides direct LLM responses without document retrieval.
    Use it for questions that don't require specific document context.
    
    Use this tool when:
    - Questions are about general knowledge topics
    - User asks for explanations of concepts
    - Query doesn't reference the uploaded document
    """
    
    name: str = "general_llm"
    description: str = (
        "Answer general knowledge questions that don't require document context. "
        "Use this tool for questions about concepts, definitions, explanations, "
        "or any topic that doesn't specifically reference the uploaded PDF. "
        "Examples: 'What is RAG?', 'Explain transformer architecture', 'Define machine learning'."
    )
    args_schema: Type[BaseModel] = GeneralLLMInput
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Return the query for the LLM to answer directly.
        
        Args:
            query: The question to answer
            run_manager: Callback manager (unused)
            
        Returns:
            The query formatted for direct LLM response
        """
        # This tool doesn't do any processing - it just signals that
        # the query should be answered directly by the LLM
        return f"Answer this general knowledge question: {query}"


def create_tools(
    vector_store_manager: Optional[VectorStoreManager] = None,
    retrieval_k: int = 4,
) -> list[BaseTool]:
    """
    Create and return the list of tools for the agent.
    
    Args:
        vector_store_manager: Manager for the vector store (optional at init)
        retrieval_k: Number of chunks to retrieve for PDF queries
        
    Returns:
        List of tool instances
    """
    pdf_tool = PDFRetrievalTool(
        vector_store_manager=vector_store_manager,
        k=retrieval_k,
    )
    
    general_tool = GeneralLLMTool()
    
    return [pdf_tool, general_tool]


# Tool name constants for easy reference
TOOL_PDF_RETRIEVAL = "pdf_retrieval"
TOOL_GENERAL_LLM = "general_llm"


def get_tool_display_name(tool_name: str) -> str:
    """
    Get a human-readable display name for a tool.
    
    Args:
        tool_name: Internal tool name
        
    Returns:
        Human-readable display name
    """
    display_names = {
        TOOL_PDF_RETRIEVAL: "PDF Retrieval",
        TOOL_GENERAL_LLM: "General LLM",
    }
    return display_names.get(tool_name, tool_name)
