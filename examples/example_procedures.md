# Example Procedural Knowledge

## How to Add New Documents to Memory

1. Navigate to the appropriate directory:
   - `semantic/` for descriptive knowledge
   - `procedural/` for how-to guides
   - `episodic/` for experience records

2. Create a new `.txt` or `.md` file with descriptive content

3. After adding documents, refresh the vector stores:
   - Ask the agent to refresh vector stores
   - Or manually run the refresh_vector_stores tool

4. The new documents will now be searchable by the agent

## How to Query Specific Memory Types

To retrieve information from a specific memory type:

1. **For conceptual understanding** - The agent will query semantic memory
   - Example: "What is a neural network?"

2. **For step-by-step instructions** - The agent will query procedural memory
   - Example: "How do I set up the development environment?"

3. **For past experiences** - The agent will query episodic memory
   - Example: "What happened last time we tried this approach?"

## How to Organize Large Document Collections

1. Create subdirectories within semantic/procedural/episodic
2. Group related documents together
3. Use clear, descriptive filenames
4. Keep individual files focused on specific topics
5. Refresh vector stores after reorganization

