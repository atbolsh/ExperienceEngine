# Architecture Overview

## Project Structure

```
experience-engine/
├── main.py                    # Entry point - user interaction loop
├── agent.py                   # Agent configuration and creation
├── vector_store.py           # FAISS vector database management
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in repo)
├── .gitignore               # Git ignore rules
│
├── tools/                    # Modular tool system
│   ├── __init__.py          # Tool aggregation and exports
│   ├── memory_tools.py      # FAISS backbone (shared)
│   ├── filesystem_tools.py  # File/directory operations
│   ├── semantic_tools.py    # Semantic memory queries
│   ├── procedural_tools.py  # Procedural memory queries
│   ├── episodic_tools.py    # Episodic memory queries
│   ├── image_tools.py       # Image retrieval and vision analysis
│   ├── memory_schema.py     # Memory folder format validation
│   ├── scripting_tools.py   # Python script management and execution
│   └── thinking_tools.py    # Sequential thinking and reasoning
│
├── semantic/                # Semantic memory documents (empty by default)
├── procedural/              # Procedural memory documents (empty by default)
├── episodic/                # Episodic memory documents (empty by default)
│
├── examples/                # Example documents for reference
│   ├── README.md
│   ├── example_concepts.md
│   ├── example_procedures.md
│   └── example_episode.md
│
└── docs/                    # Documentation
    ├── README.md
    ├── QUICKSTART.md
    └── ARCHITECTURE.md (this file)
```

## Component Responsibilities

### Core Components

#### `main.py`
- **Purpose**: Entry point and user interaction loop
- **Responsibilities**:
  - Load environment variables
  - Create the conversational agent
  - Handle user input/output
  - Manage conversation flow (exit, clear, etc.)
- **Dependencies**: `agent.py`, `dotenv`

#### `agent.py`
- **Purpose**: Agent configuration and creation
- **Responsibilities**:
  - Initialize vector store manager
  - Configure LLM (GPT-4o)
  - Create prompt templates
  - Build AgentExecutor with tools
  - Wrap agent with conversation memory
- **Dependencies**: `tools`, `langchain`, `langchain_openai`

#### `vector_store.py`
- **Purpose**: FAISS vector database management
- **Responsibilities**:
  - Load documents from directories
  - Create/load FAISS indices
  - Chunk documents for embedding
  - Query vector stores
  - Refresh indices when documents change
- **Dependencies**: `langchain`, `faiss`, `openai`

### Tools System

The tools are organized in a modular fashion, making it easy to add new capabilities:

#### `tools/__init__.py`
- **Purpose**: Tool aggregation
- **Responsibilities**:
  - Import all tool modules
  - Provide `create_tools()` function
  - Export common functions
- **Pattern**: Central registry for all tools

#### `tools/memory_tools.py`
- **Purpose**: FAISS backbone (shared infrastructure)
- **Responsibilities**:
  - Initialize vector store manager
  - Provide `query_memory_store()` function
  - Provide `refresh_all_vector_stores()` function
- **Pattern**: Shared utility functions for memory operations

#### `tools/filesystem_tools.py`
- **Purpose**: File and directory operations
- **Tools Provided**:
  - `list_directory`
  - `read_file`
  - `write_file`
  - `append_file`
  - `delete_file`
  - `create_directory`
  - `get_current_directory`
- **Pattern**: Self-contained tool module

#### `tools/semantic_tools.py`
- **Purpose**: Semantic memory queries
- **Tools Provided**:
  - `query_semantic_memory`
- **Dependencies**: `tools.memory_tools`
- **Pattern**: Thin wrapper around memory backbone

#### `tools/procedural_tools.py`
- **Purpose**: Procedural memory queries
- **Tools Provided**:
  - `query_procedural_memory`
- **Dependencies**: `tools.memory_tools`
- **Pattern**: Thin wrapper around memory backbone

#### `tools/episodic_tools.py`
- **Purpose**: Episodic memory queries
- **Tools Provided**:
  - `query_episodic_memory`
- **Dependencies**: `tools.memory_tools`
- **Pattern**: Thin wrapper around memory backbone

#### `tools/image_tools.py`
- **Purpose**: Image retrieval and vision analysis from memory folders
- **Tools Provided**:
  - `list_memory_images` - List available images in a memory
  - `retrieve_memory_image` - Get image information
  - `analyze_memory_image` - Use GPT-4o vision to analyze images in context
- **Dependencies**: `langchain_openai`, `tools.memory_schema`, `base64`
- **Pattern**: Vision-enabled tool module
- **Features**:
  - On-demand image access (not loaded during indexing)
  - Context-aware analysis (includes memory text)
  - Supports multiple image formats

#### `tools/memory_schema.py`
- **Purpose**: Memory folder format definition and validation
- **Functionality**:
  - Defines MemoryFolder dataclass
  - Validates folder structure (1 text file, 0-10 images)
  - Loads memory folders with metadata
  - Lists all valid memories in a directory
- **Dependencies**: `pathlib`, `dataclasses`
- **Pattern**: Shared utility module

#### `tools/scripting_tools.py`
- **Purpose**: Python script management and execution
- **Tools Provided**:
  - `write_script` - Create Python scripts in scripts/ directory
  - `execute_script` - Run scripts with optional arguments (30s timeout)
  - `list_scripts` - List available scripts
  - `read_script` - Read script contents
  - `delete_script` - Remove scripts
- **Dependencies**: `subprocess`, `pathlib`
- **Pattern**: Self-contained tool module with security features
- **Security Features**:
  - Path traversal prevention (blocks `..`, `/`, `\`)
  - Execution timeout (30 seconds)
  - Scripts isolated to scripts/ directory
  - Automatic .py extension handling

#### `tools/thinking_tools.py`
- **Purpose**: Sequential thinking and step-by-step reasoning
- **Tools Provided**:
  - `SequentialThinkingTool` (from sequential-thinking-tool package)
- **Dependencies**: `sequential-thinking-tool`
- **Pattern**: Wrapper around external tool package
- **Features**:
  - Break down complex problems into steps
  - Track thought progression
  - Maintain thinking history

## Data Flow

### Initialization Flow
```
main.py
  └─> load_dotenv()
  └─> agent.create_conversational_agent()
        └─> agent.create_agent()
              └─> tools.initialize_vector_store_manager()
                    └─> vector_store.VectorStoreManager()
                          └─> Load/create FAISS indices for each memory type
              └─> tools.create_tools()
                    └─> Aggregate all tools from modules
              └─> Create AgentExecutor with LLM and tools
        └─> Wrap with RunnableWithMessageHistory
```

### Query Flow
```
User Input
  └─> main.py receives input
        └─> agent.invoke(input)
              └─> AgentExecutor processes with LLM
                    └─> LLM decides which tools to use
                          └─> Tool execution
                                ├─> Filesystem tools (direct operations)
                                ├─> Scripting tools (create/execute scripts)
                                ├─> Thinking tools (sequential reasoning)
                                └─> Memory tools
                                      └─> memory_tools.query_memory_store()
                                            └─> vector_store.query_store()
                                                  └─> FAISS similarity search
              └─> Return response
        └─> Display to user
```

## Design Principles

### 1. Modularity
- Each tool category in its own file
- Easy to add new tool files
- Clear separation of concerns

### 2. Reusability
- `memory_tools.py` provides shared FAISS functionality
- Semantic/procedural/episodic tools reuse the same backbone
- Avoids code duplication

### 3. Extensibility
To add new tools:
1. Create `tools/new_tools.py`
2. Define functions and schemas
3. Create `create_new_tools()` function
4. Add to `tools/__init__.py`

### 4. Clean Separation
- **main.py**: UI/interaction
- **agent.py**: Agent configuration
- **vector_store.py**: Data management
- **tools/**: Capabilities

## Memory System

### Three Types of Memory

Each memory is a **folder** containing one text file and optionally 0-10 images.

1. **Semantic Memory** (`semantic/`)
   - Descriptive information
   - Concepts and definitions
   - General knowledge
   - "What" something is
   - Images: diagrams, charts, concept illustrations

2. **Procedural Memory** (`procedural/`)
   - Step-by-step instructions
   - How-to guides
   - Procedures and workflows
   - "How" to do something
   - Images: screenshots, step visualizations, examples

3. **Episodic Memory** (`episodic/`)
   - Past interactions
   - Specific experiences
   - Temporal events
   - "When" something happened
   - Images: photos, state captures, observations

### Vector Store Implementation

- **Technology**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: OpenAI embeddings
- **Chunking**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Persistence**: Stored in `.faiss_*` directories
- **Formats**: Supports `.txt` and `.md` files

## Extension Points

### Adding New Memory Types
1. Create new directory (e.g., `spatial/`)
2. Update `vector_store.py` to handle new type
3. Create `tools/spatial_tools.py`
4. Add to `tools/__init__.py`

### Adding New Tool Categories
1. Create `tools/category_tools.py`
2. Define tools following existing patterns
3. Add to `create_tools()` in `tools/__init__.py`

### Customizing Agent Behavior
- Modify system prompt in `agent.py`
- Adjust temperature, max_iterations
- Change LLM model

## Future Enhancements

Potential areas for expansion:
- Environment interaction tools (as mentioned in original goal)
- Multi-session persistence
- Memory consolidation strategies
- Custom embedding models
- Metadata filtering in searches
- Advanced chunking strategies
- Tool usage analytics
- Memory importance scoring

