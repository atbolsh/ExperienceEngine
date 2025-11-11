# Experience Engine Agent

An AI agent system that explores environments, records experiences, and leverages saved information across sessions. The agent features RAG (Retrieval-Augmented Generation) capabilities with three types of memory stores and comprehensive filesystem manipulation tools.

## Features

### 🧠 Three Types of Memory
- **Semantic Memory**: Descriptive information, concepts, and definitions about the environment
- **Procedural Memory**: Step-by-step instructions for common interactions and tasks
- **Episodic Memory**: Detailed, truthful records of instructive prior interactions

### 🔧 Filesystem Tools
- List directory contents
- Read and write files
- Append to files
- Create and delete directories
- Delete files
- Get current working directory

### 🧠 Sequential Thinking
- Break down complex problems into step-by-step reasoning
- Organize multi-step thought processes
- Track thinking history and progression

### 💬 Conversational Memory
- Maintains context across the conversation
- References previous exchanges
- Clear conversation history with the `clear` command

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /path/to/experience-engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```
   
   Get your API key from: https://platform.openai.com/api-keys

## Usage

### Running the Agent

```bash
python main.py
```

### Commands

- Type your queries naturally to interact with the agent
- Type `exit`, `quit`, or `q` to end the session
- Type `clear` to clear conversation history

### Example Interactions

```
You: What files are in the current directory?
Assistant: [Uses list_directory tool]

You: Create a new directory called "notes"
Assistant: [Uses create_directory tool]

You: Write "Hello World" to notes/greeting.txt
Assistant: [Uses write_file tool]

You: What do you know about navigating the environment?
Assistant: [May query procedural memory if relevant]
```

## Memory System

### Adding Documents

The agent can read from three directories for long-term memory:

1. **semantic/** - Add `.txt` or `.md` files with descriptive information
2. **procedural/** - Add `.txt` or `.md` files with step-by-step guides
3. **episodic/** - Add `.txt` or `.md` files with detailed interaction records

### Refreshing Vector Stores

After adding new documents, ask the agent to refresh the vector stores:

```
You: Please refresh the vector stores
Assistant: [Uses refresh_vector_stores tool]
```

Or the agent can do it automatically when appropriate.

### How Memory Works

- Each directory has its own FAISS vector index (stored in `.faiss_*` directories)
- Documents are automatically chunked and embedded using OpenAI embeddings
- The agent queries these stores ONLY when relevant to the user's question
- Vector stores are created on first run and loaded from disk on subsequent runs

## Architecture

### Files

- **main.py** - Main entry point with user interaction loop
- **agent.py** - Agent creation and configuration
- **vector_store.py** - FAISS vector store management and document loading
- **tools/** - Modular tool definitions
  - **__init__.py** - Tool aggregation and exports
  - **filesystem_tools.py** - File and directory operations
  - **memory_tools.py** - FAISS backbone for memory queries
  - **semantic_tools.py** - Semantic memory queries
  - **procedural_tools.py** - Procedural memory queries
  - **episodic_tools.py** - Episodic memory queries
  - **thinking_tools.py** - Sequential thinking and reasoning tools
- **requirements.txt** - Python dependencies
- **.env** - Environment variables (API keys)
- **examples/** - Example documents for reference

### Key Components

- **LLM**: GPT-4o via LangChain
- **Agent**: LangChain AgentExecutor with tool calling
- **Memory**: InMemoryChatMessageHistory for conversation context
- **Vector Store**: FAISS with OpenAI embeddings
- **Tools**: Custom LangChain tools for filesystem and RAG operations

## Advanced Features

### Filesystem Operations

The agent can perform complex filesystem tasks:
- Read configuration files
- Write logs or outputs
- Organize files into directories
- Clean up temporary files

### RAG Integration

The agent intelligently decides when to query memory stores:
- Queries semantic memory for conceptual understanding
- Queries procedural memory for how-to instructions
- Queries episodic memory for past experiences

### Extensibility

Easy to extend with additional tools:
1. Create a new file in the `tools/` directory (e.g., `tools/custom_tools.py`)
2. Define your tool functions and input schemas using Pydantic
3. Create a `create_custom_tools()` function that returns a list of Tool objects
4. Import and add your tools in `tools/__init__.py` in the `create_tools()` function
5. The agent will automatically learn to use them

Example structure for a new tool file:
```python
# tools/custom_tools.py
from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field

class CustomInput(BaseModel):
    param: str = Field(description="Description")

def custom_function(param: str) -> str:
    return f"Result: {param}"

def create_custom_tools():
    return [
        Tool(
            name="custom_tool",
            func=custom_function,
            description="Tool description",
            args_schema=CustomInput,
        ),
    ]
```

## Tips

1. **Be specific**: Clear queries get better results
2. **Use memory wisely**: Add high-quality documents to the memory directories
3. **Refresh after updates**: Remember to refresh vector stores after adding documents
4. **Organize documents**: Use subdirectories in semantic/procedural/episodic for organization
5. **Monitor token usage**: GPT-4o queries cost money, so be mindful of extensive operations

## Future Enhancements

This agent is designed as a foundation for:
- Environment exploration tools (coming soon)
- Multi-session learning and adaptation
- More sophisticated memory consolidation
- Custom embedding models
- Enhanced retrieval strategies

## Troubleshooting

### "Vector store manager not initialized"
- Ensure the agent has properly started
- Check that semantic/procedural/episodic directories exist

### "No relevant information found"
- Add documents to the appropriate directory
- Use the refresh_vector_stores tool
- Check that documents are in `.txt` or `.md` format

### FAISS import errors
- Ensure `faiss-cpu` is installed: `pip install faiss-cpu`
- On some systems you may need `faiss-gpu` instead

### OpenAI API errors
- Verify your API key in `.env`
- Check your OpenAI account has credits
- Ensure you have access to GPT-4o

## License

This project is designed for experimentation and learning with AI agents.

