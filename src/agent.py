"""
Agent Module.

Creates and manages the LangChain agent with intelligent tool routing.
Handles conversation memory and orchestrates responses between the two tools.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from src.tools import (
    create_tools,
    PDFRetrievalTool,
    TOOL_PDF_RETRIEVAL,
    TOOL_GENERAL_LLM,
    get_tool_display_name,
)
from src.vector_store import VectorStoreManager


# System prompt for the agent
SYSTEM_PROMPT = """You are an intelligent PDF Question-Answering Assistant with access to two tools:

1. **PDF Retrieval Tool** (`pdf_retrieval`): Use this when the user's question is about content in the uploaded PDF document. This tool searches the document and returns relevant passages.

2. **General LLM Tool** (`general_llm`): Use this for general knowledge questions that don't require the PDF document context.

## Tool Selection Guidelines:

**Use PDF Retrieval when:**
- The question asks about specific content from the uploaded document
- The user references "the document", "the PDF", "this paper", etc.
- Questions about figures, tables, or sections mentioned in the document
- The query requires information that would be in the uploaded PDF

**Use General LLM when:**
- The question is about general concepts or definitions
- The user asks for explanations not specific to the document
- Questions like "What is RAG?", "Explain transformers", etc.
- The query doesn't reference the uploaded document

## Response Guidelines:
- Always provide clear, accurate, and helpful responses
- When using PDF retrieval, cite the page numbers from the retrieved context
- If the PDF retrieval doesn't find relevant information, say so honestly
- For general questions, provide comprehensive explanations
- Be conversational and maintain context from previous messages

Current date: {current_date}
"""


class PDFAgent:
    """
    PDF Question-Answering Agent with intelligent tool routing.
    
    Manages:
    - LLM initialization (Ollama)
    - Tool creation and management
    - Conversation history (simple list-based)
    - Agent execution
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        vector_store_manager: Optional[VectorStoreManager] = None,
        retrieval_k: int = 4,
    ):
        """
        Initialize the PDF Agent.
        
        Args:
            model_name: Name of the Ollama model to use
            ollama_base_url: URL of the Ollama server
            temperature: Temperature for LLM responses
            vector_store_manager: Manager for PDF vector store
            retrieval_k: Number of chunks to retrieve for PDF queries
        """
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.vector_store_manager = vector_store_manager
        self.retrieval_k = retrieval_k
        
        # Initialize components
        self._llm: Optional[ChatOllama] = None
        self._tools = None
        self._agent_executor: Optional[AgentExecutor] = None
        
        # Simple list-based conversation history
        self._chat_history: List[BaseMessage] = []
        
        # Track last used tool for display
        self._last_tool_used: Optional[str] = None
    
    @property
    def llm(self) -> ChatOllama:
        """Lazy-load the LLM."""
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.model_name,
                base_url=self.ollama_base_url,
                temperature=self.temperature,
            )
        return self._llm
    
    @property
    def tools(self) -> List:
        """Get the tools list."""
        if self._tools is None:
            self._tools = create_tools(
                vector_store_manager=self.vector_store_manager,
                retrieval_k=self.retrieval_k,
            )
        return self._tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the tool-calling agent."""
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        # Create the agent executor (no memory - we manage history manually)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,  # Set to False in production
            handle_parsing_errors=True,
            max_iterations=5,
        )
        
        return agent_executor
    
    @property
    def agent_executor(self) -> AgentExecutor:
        """Get or create the agent executor."""
        if self._agent_executor is None:
            self._agent_executor = self._create_agent()
        return self._agent_executor
    
    def update_vector_store(self, vector_store_manager: VectorStoreManager) -> None:
        """
        Update the vector store manager and recreate tools.
        
        Args:
            vector_store_manager: New vector store manager
        """
        self.vector_store_manager = vector_store_manager
        
        # Recreate tools with new vector store
        self._tools = create_tools(
            vector_store_manager=vector_store_manager,
            retrieval_k=self.retrieval_k,
        )
        
        # Recreate agent with new tools
        self._agent_executor = None
    
    def clear_memory(self) -> None:
        """Clear the conversation history."""
        self._chat_history = []
        self._last_tool_used = None
    
    def invoke(self, query: str) -> Tuple[str, str]:
        """
        Process a user query and return the response.
        
        Args:
            query: User's question
            
        Returns:
            Tuple of (response text, tool used name)
        """
        try:
            # Invoke the agent with current chat history
            result = self.agent_executor.invoke({
                "input": query,
                "chat_history": self._chat_history,
                "current_date": datetime.now().strftime("%Y-%m-%d"),
            })
            
            # Extract the response
            response = result.get("output", "I couldn't process your request.")
            
            # Update chat history with this exchange
            self._chat_history.append(HumanMessage(content=query))
            self._chat_history.append(AIMessage(content=response))
            
            # Determine which tool was used by checking intermediate steps
            tool_used = self._determine_tool_used(result)
            self._last_tool_used = tool_used
            
            return response, tool_used
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            return error_msg, "error"
    
    def _determine_tool_used(self, result: Dict[str, Any]) -> str:
        """
        Determine which tool was used based on agent output.
        
        Args:
            result: Agent execution result
            
        Returns:
            Tool name that was used
        """
        # Check intermediate steps for tool calls
        intermediate_steps = result.get("intermediate_steps", [])
        
        if intermediate_steps:
            for step in intermediate_steps:
                if len(step) >= 1:
                    action = step[0]
                    if hasattr(action, 'tool'):
                        return action.tool
        
        # Fallback: check if PDF retrieval was attempted
        output = result.get("output", "").lower()
        if "pdf" in output or "document" in output or "page" in output:
            return TOOL_PDF_RETRIEVAL
        
        return TOOL_GENERAL_LLM
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get the chat history as a list of message dictionaries.
        
        Returns:
            List of {"role": "user/assistant", "content": "..."} dicts
        """
        history = []
        
        for msg in self._chat_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        
        return history
    
    def stream(self, query: str):
        """
        Stream the agent's response.
        
        Note: This is a simplified streaming implementation.
        Full streaming with tool calls requires more complex handling.
        
        Args:
            query: User's question
            
        Yields:
            Chunks of the response
        """
        # For now, use non-streaming and yield the full response
        # Full streaming with tool calls is complex in LangChain
        response, tool_used = self.invoke(query)
        yield response, tool_used


def create_agent(
    model_name: str = "llama3.2",
    ollama_base_url: str = "http://localhost:11434",
    vector_store_manager: Optional[VectorStoreManager] = None,
) -> PDFAgent:
    """
    Factory function to create a PDF Agent.
    
    Args:
        model_name: Name of the Ollama model
        ollama_base_url: URL of the Ollama server
        vector_store_manager: Optional vector store manager
        
    Returns:
        Configured PDFAgent instance
    """
    return PDFAgent(
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        vector_store_manager=vector_store_manager,
    )
