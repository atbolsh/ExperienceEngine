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
✅ **Three memory types**: semantic, procedural, episodic (with images!)  
✅ **FAISS vector stores** for intelligent text retrieval  
✅ **Vision-enabled**: GPT-4o analyzes images from memories  
✅ **Filesystem tools**: read, write, list, create, delete  
✅ **Scripting tools**: write and execute Python scripts  
✅ **Sequential thinking**: step-by-step reasoning for complex problems  
✅ **Chat memory**: maintains context across conversation  
✅ **Smart RAG**: only queries memory when relevant  

## Next Steps

1. Check out the example memory folders in `examples/`
2. Read `examples/MEMORY_FORMAT.md` for the memory folder structure
3. Create your own memory folders in semantic/procedural/episodic:
   ```
   semantic/my_concept/
   ├── description.txt
   ├── diagram.png
   └── example.jpg
   ```
4. Ask the agent to refresh vector stores after adding memories
5. Try asking questions that reference images in memories
6. Experiment with filesystem operations and scripting
7. Clear history with the `clear` command when needed

Enjoy exploring! 🚀

