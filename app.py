"""
PDF Question-Answering Agent with Streamlit Interface.

A production-ready RAG application that intelligently routes queries between
PDF document retrieval and general LLM knowledge.

Usage:
    streamlit run app.py
    
Requirements:
    - Ollama running locally with a model (e.g., llama3.2)
    - Python 3.11+
"""

import os
import streamlit as st
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.pdf_processor import PDFProcessor, get_pdf_stats
from src.vector_store import VectorStoreManager
from src.agent import PDFAgent
from src.tools import get_tool_display_name, TOOL_PDF_RETRIEVAL


# =============================================================================
# Configuration
# =============================================================================

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="PDF QA Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration constants
APP_TITLE = "PDF Question-Answering Agent"
APP_ICON = "📄"
DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_FILE_SIZE_MB = 50
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4


# =============================================================================
# Session State Management
# =============================================================================

def init_session_state() -> None:
    """Initialize all session state variables."""
    
    # Chat history: List of {"role": "user/assistant", "content": str, "tool": str, "timestamp": str}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # PDF processing state
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    
    if "pdf_filename" not in st.session_state:
        st.session_state.pdf_filename = None
    
    if "pdf_page_count" not in st.session_state:
        st.session_state.pdf_page_count = 0
    
    if "pdf_chunk_count" not in st.session_state:
        st.session_state.pdf_chunk_count = 0
    
    # Vector store manager
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None
    
    # PDF processor
    if "pdf_processor" not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    
    # Agent
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    # Processing flags
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False


def clear_chat() -> None:
    """Clear the chat history and reset conversation memory."""
    st.session_state.chat_history = []
    
    # Clear agent memory if agent exists
    if st.session_state.agent is not None:
        st.session_state.agent.clear_memory()


def reset_pdf_state() -> None:
    """Reset all PDF-related state."""
    st.session_state.pdf_uploaded = False
    st.session_state.pdf_filename = None
    st.session_state.pdf_page_count = 0
    st.session_state.pdf_chunk_count = 0
    
    # Clear vector store
    if st.session_state.vector_store_manager is not None:
        st.session_state.vector_store_manager.delete_collection()
    
    st.session_state.vector_store_manager = None
    
    # Reset agent's vector store reference
    if st.session_state.agent is not None:
        st.session_state.agent.update_vector_store(None)
    
    # Also clear chat when PDF changes
    clear_chat()


# =============================================================================
# PDF Processing Functions
# =============================================================================

def process_uploaded_pdf(uploaded_file: BytesIO, filename: str) -> Dict[str, Any]:
    """
    Process an uploaded PDF file.
    
    Args:
        uploaded_file: BytesIO object containing PDF data
        filename: Name of the uploaded file
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Process PDF
        processor = st.session_state.pdf_processor
        chunks, metadata = processor.process_pdf(uploaded_file, filename)
        
        # Create vector store
        vector_store_manager = VectorStoreManager(
            persist_directory=f"./chroma_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        vector_store_manager.create_vector_store(chunks, persist=False)
        
        # Update session state
        st.session_state.vector_store_manager = vector_store_manager
        st.session_state.pdf_uploaded = True
        st.session_state.pdf_filename = filename
        st.session_state.pdf_page_count = metadata["page_count"]
        st.session_state.pdf_chunk_count = metadata["chunk_count"]
        
        # Update or create agent with vector store
        if st.session_state.agent is None:
            st.session_state.agent = PDFAgent(
                model_name=DEFAULT_MODEL,
                ollama_base_url=OLLAMA_BASE_URL,
                vector_store_manager=vector_store_manager,
                retrieval_k=RETRIEVAL_K,
            )
        else:
            st.session_state.agent.update_vector_store(vector_store_manager)
        
        return {
            "success": True,
            "page_count": metadata["page_count"],
            "chunk_count": metadata["chunk_count"],
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar() -> Optional[BytesIO]:
    """
    Render the sidebar with PDF upload and controls.
    
    Returns:
        Uploaded file BytesIO if a file was uploaded, None otherwise
    """
    with st.sidebar:
        st.header("📄 PDF Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB",
            key="pdf_uploader",
        )
        
        # Display current PDF info if uploaded
        if st.session_state.pdf_uploaded:
            st.success(f"✅ PDF Loaded: {st.session_state.pdf_filename}")
            st.info(f"📖 Pages: {st.session_state.pdf_page_count}")
            st.info(f"📦 Chunks: {st.session_state.pdf_chunk_count}")
        
        st.divider()
        
        # Clear chat button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_chat()
                st.rerun()
        
        with col2:
            if st.button("🔄 New PDF", use_container_width=True):
                reset_pdf_state()
                st.rerun()
        
        st.divider()
        
        # Model settings
        st.subheader("⚙️ Settings")
        
        model_name = st.text_input(
            "Ollama Model",
            value=DEFAULT_MODEL,
            help="Name of the Ollama model to use",
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random",
        )
        
        # Update agent settings if changed
        if st.session_state.agent is not None:
            if model_name != st.session_state.agent.model_name:
                st.session_state.agent = PDFAgent(
                    model_name=model_name,
                    ollama_base_url=OLLAMA_BASE_URL,
                    temperature=temperature,
                    vector_store_manager=st.session_state.vector_store_manager,
                )
        
        st.divider()
        
        # Instructions
        st.markdown("""
        ### 📋 Instructions
        
        1. **Upload a PDF** using the file uploader above
        2. **Wait** for processing to complete
        3. **Ask questions** about the document or general topics
        4. The agent will automatically choose the right tool
        
        ### 🔧 Requirements
        
        - Ollama must be running locally
        - Pull a model: `ollama pull llama3.2`
        - Start Ollama: `ollama serve`
        """)
        
        return uploaded_file


def render_chat_message(message: Dict[str, Any]) -> None:
    """
    Render a single chat message.
    
    Args:
        message: Message dictionary with role, content, tool, timestamp
    """
    role = message["role"]
    content = message["content"]
    tool = message.get("tool")
    timestamp = message.get("timestamp", "")
    
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
            if timestamp:
                st.caption(f"⏰ {timestamp}")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            
            # Show tool used
            if tool:
                tool_display = get_tool_display_name(tool)
                if tool == "pdf_retrieval":
                    st.caption(f"🔧 [Used: {tool_display}]")
                else:
                    st.caption(f"🔧 [Used: {tool_display}]")
            
            if timestamp:
                st.caption(f"⏰ {timestamp}")


def render_chat_interface() -> None:
    """Render the main chat interface."""
    st.title(APP_TITLE)
    st.markdown("""
    Ask questions about your uploaded PDF document or general knowledge topics.
    The agent will intelligently route your query to the appropriate tool.
    """)
    
    # Display chat history
    for message in st.session_state.chat_history:
        render_chat_message(message)
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to history
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.chat_history.append(user_message)
        
        # Render user message immediately
        render_chat_message(user_message)
        
        # Check if agent is available
        if st.session_state.agent is None:
            # Create agent without vector store for general questions
            st.session_state.agent = PDFAgent(
                model_name=DEFAULT_MODEL,
                ollama_base_url=OLLAMA_BASE_URL,
            )
        
        # Process query with loading indicator
        with st.spinner("Thinking..."):
            try:
                response, tool_used = st.session_state.agent.invoke(prompt)
            except Exception as e:
                response = f"Error: {str(e)}. Please ensure Ollama is running with the model '{DEFAULT_MODEL}'."
                tool_used = "error"
        
        # Add assistant message to history
        assistant_message = {
            "role": "assistant",
            "content": response,
            "tool": tool_used,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.chat_history.append(assistant_message)
        
        # Render assistant message
        render_chat_message(assistant_message)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    uploaded_file = render_sidebar()
    
    # Handle file upload
    if uploaded_file is not None:
        # Check if this is a new file
        current_filename = uploaded_file.name
        if current_filename != st.session_state.pdf_filename:
            # Reset previous state
            reset_pdf_state()
            
            # Process new file
            with st.spinner("Processing PDF..."):
                # Read file content
                pdf_bytes = BytesIO(uploaded_file.getvalue())
                result = process_uploaded_pdf(pdf_bytes, current_filename)
            
            if result["success"]:
                st.success(f"✅ PDF processed successfully! ({result['page_count']} pages, {result['chunk_count']} chunks)")
            else:
                st.error(f"❌ Error processing PDF: {result.get('error', 'Unknown error')}")
    
    # Render chat interface
    render_chat_interface()


if __name__ == "__main__":
    main()
