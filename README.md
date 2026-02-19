# PDF Question-Answering Agent

A production-ready RAG (Retrieval-Augmented Generation) application that intelligently routes user queries between PDF document retrieval and general LLM knowledge.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local-purple.svg)

## Features

- **Intelligent Tool Routing**: Automatically routes queries between PDF retrieval and general LLM knowledge
- **PDF Processing**: Upload and process PDF documents with automatic chunking
- **Semantic Search**: Uses HuggingFace embeddings and ChromaDB for efficient similarity search
- **Conversational Memory**: Maintains context across multiple questions
- **Modern UI**: Clean Streamlit interface with chat-style layout
- **Local LLM**: Uses Ollama for privacy-preserving local inference

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Interface                      │
│  ┌─────────────┐  ┌──────────────────────────────────────┐  │
│  │   Sidebar   │  │           Main Chat Area             │  │
│  │  - PDF      │  │  - User messages                     │  │
│  │    Upload   │  │  - Agent responses                   │  │
│  │  - Clear    │  │  - Tool invocation labels            │  │
│  │    Chat     │  │  - Timestamps                        │  │
│  └─────────────┘  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangChain Agent                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Tool Selection Logic                     │   │
│  │   - Analyzes query semantics                          │   │
│  │   - Routes to appropriate tool                        │   │
│  │   - Maintains conversation memory                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│              ┌───────────────┴───────────────┐               │
│              ▼                               ▼               │
│  ┌─────────────────────┐       ┌─────────────────────┐      │
│  │  PDF Retrieval Tool │       │   General LLM Tool  │      │
│  │  - Vector search    │       │  - Direct response  │      │
│  │  - Context retrieval│       │  - No retrieval     │      │
│  └─────────────────────┘       └─────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Vector Store (ChromaDB)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PDF Chunks + HuggingFace Embeddings                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
infinijith/
├── app.py                 # Main Streamlit application
├── main.py                # Entry point (alternative)
├── pyproject.toml         # Project dependencies
├── README.md              # This file
└── src/
    ├── __init__.py        # Package initialization
    ├── pdf_processor.py   # PDF loading and chunking
    ├── vector_store.py    # ChromaDB vector store management
    ├── tools.py           # PDF Retrieval and General LLM tools
    └── agent.py           # LangChain agent with tool routing
```

## Prerequisites

1. **Python 3.11+**: Ensure Python is installed
2. **Ollama**: Install and run Ollama for local LLM inference

## Installation

### 1. Install Ollama

**Windows:**
```bash
# Download from https://ollama.ai/download
# Or use winget
winget install Ollama.Ollama
```

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull the LLM Model

```bash
# Pull the default model (llama3.2)
ollama pull llama3.2

# Alternative models
ollama pull mistral
ollama pull codellama
```

### 3. Start Ollama Server

```bash
ollama serve
```

### 4. Install Python Dependencies

Using `uv` (recommended):
```bash
# Install uv if not already installed
pip install uv

# Install dependencies
uv sync
```

Using `pip`:
```bash
pip install -e .
```

## Usage

### Running the Application

```bash
# Using Streamlit directly
streamlit run app.py

# Or using uv
uv run streamlit run app.py
```

### Using the Interface

1. **Upload a PDF**: Use the sidebar file uploader to upload a PDF document
2. **Wait for Processing**: The PDF will be automatically processed and indexed
3. **Ask Questions**: Type questions in the chat input
   - Document-specific questions will use PDF Retrieval
   - General questions will use the General LLM tool
4. **View Tool Usage**: Each response shows which tool was used

### Example Questions

**PDF Retrieval (document-specific):**
- "What is the main topic of this document?"
- "Summarize the key findings from section 3"
- "What does the document say about [topic]?"

**General LLM (knowledge questions):**
- "What is RAG?"
- "Explain transformer architecture"
- "What are the benefits of vector databases?"

## Configuration

### Environment Variables

Create a `.env` file for custom configuration:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4
```

### Model Settings

Adjust settings in the Streamlit sidebar:
- **Model Name**: Change the Ollama model
- **Temperature**: Control response randomness (0.0-1.0)

## Technical Details

### PDF Processing

- Uses `PyPDF` for PDF parsing
- `RecursiveCharacterTextSplitter` for chunking
- Default chunk size: 1000 characters
- Default overlap: 200 characters

### Vector Store

- **Database**: ChromaDB (in-memory or persisted)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (free, local)
- **Similarity Search**: Cosine similarity

### Agent Architecture

- **Framework**: LangChain Agent with tool calling
- **Memory**: `ConversationBufferMemory`
- **Tool Selection**: Automatic based on query analysis

## Troubleshooting

### Ollama Connection Error

```
Error: Cannot connect to Ollama server
```

**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

### Model Not Found

```
Error: Model 'llama3.2' not found
```

**Solution**: Pull the model:
```bash
ollama pull llama3.2
```

### PDF Processing Error

```
Error: Failed to load PDF
```

**Solutions**:
- Ensure the PDF is not password-protected
- Check file size (max 50MB)
- Try a different PDF file

### Memory Issues

For large PDFs, you may need to adjust chunk settings:

```python
# In app.py, modify these constants
CHUNK_SIZE = 500      # Smaller chunks
CHUNK_OVERLAP = 100   # Less overlap
```

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --dev

# Run tests
pytest tests/
```

### Code Style

```bash
# Format code
ruff format .

# Lint code
ruff check .
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Agent framework
- [Streamlit](https://streamlit.io/) - Web interface
- [Ollama](https://ollama.ai/) - Local LLM inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [HuggingFace](https://huggingface.co/) - Embeddings
