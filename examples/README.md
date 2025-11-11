# Example Memory Folders

This directory contains example memory folders that demonstrate the new folder-based memory structure.

## Memory Folder Format

**Important:** Memories are now stored as **folders**, not individual files!

Each memory folder contains:
- **Exactly ONE text file** (`.txt` or `.md`) - The main content
- **0-10 images** (optional) - Visual content with descriptive filenames

See `MEMORY_FORMAT.md` for complete documentation.

## Example Folders

- `example_semantic_memory/` - Example of semantic memory (concepts with images)
- `example_procedural_memory/` - Example of procedural memory (how-to with screenshots)
- `example_episodic_memory/` - Example of episodic memory (experiences with photos)

Each folder contains:
- One text file with the memory content
- Placeholder files explaining what images could be included

## Usage

To add memories to your agent:

1. **Create a memory folder** in the appropriate directory:
   ```
   semantic/my_concept/
   procedural/my_procedure/
   episodic/my_experience/
   ```

2. **Add exactly one text file**:
   ```
   semantic/my_concept/description.txt
   ```

3. **Optionally add images (0-10)**:
   ```
   semantic/my_concept/diagram.png
   semantic/my_concept/example.jpg
   ```

4. **Refresh vector stores**:
   Ask the agent to refresh, or use the `refresh_vector_stores` tool.

## Image Access

- Text content is automatically indexed in FAISS
- Images are accessed on-demand using image tools:
  - `list_memory_images` - See available images
  - `analyze_memory_image` - Analyze images with GPT-4o vision

## Note

The actual memory directories (`semantic/`, `procedural/`, `episodic/`) should only contain real memories, not these examples. These are kept here for reference.

