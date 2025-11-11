# Quick Start Guide

## Setup (5 minutes)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key**
   
   Create a `.env` file:
   ```bash
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```
   
   Replace `your_key_here` with your actual OpenAI API key from https://platform.openai.com/api-keys

3. **Run the agent**
   ```bash
   python main.py
   ```

## Your First Conversation

```
You: list the current directory
Assistant: [Shows files and folders]

You: what files are in the semantic directory?
Assistant: [Lists semantic memory files]

You: what do you know about vector databases?
Assistant: [Queries semantic memory and provides information]

You: how do I add new documents?
Assistant: [Queries procedural memory for instructions]
```

## What Just Got Created

```
experience-engine/
├── main.py              # Agent entry point
├── agent.py             # Agent configuration
├── vector_store.py      # FAISS management
├── requirements.txt     # Dependencies
├── README.md           # Full documentation
├── tools/              # Modular tool definitions
│   ├── __init__.py
│   ├── filesystem_tools.py
│   ├── memory_tools.py
│   ├── semantic_tools.py
│   ├── procedural_tools.py
│   ├── episodic_tools.py
│   ├── scripting_tools.py
│   └── thinking_tools.py
├── semantic/           # Descriptive knowledge (empty)
├── procedural/         # How-to guides (empty)
├── episodic/           # Experience records (empty)
└── examples/           # Example documents
    ├── example_concepts.md
    ├── example_procedures.md
    └── example_episode.md
```

## Key Features

✅ **Conversational agent** with GPT-4o  
✅ **Three memory types**: semantic, procedural, episodic  
✅ **FAISS vector stores** for intelligent retrieval  
✅ **Filesystem tools**: read, write, list, create, delete  
✅ **Scripting tools**: write and execute Python scripts  
✅ **Sequential thinking**: step-by-step reasoning for complex problems  
✅ **Chat memory**: maintains context across conversation  
✅ **Smart RAG**: only queries memory when relevant  

## Next Steps

1. Try asking questions that use the example documents
2. Add your own documents to semantic/procedural/episodic
3. Ask the agent to refresh vector stores after adding documents
4. Experiment with filesystem operations
5. Clear history with the `clear` command when needed

Enjoy exploring! 🚀

