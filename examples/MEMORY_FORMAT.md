# Memory Folder Format

## Overview

Memories in this system are stored as **folders**, not individual files. This allows each memory to include both text content and relevant images.

## Folder Structure

Each memory folder must follow this schema:

```
memory_name/
├── description.txt (or any .txt/.md file - EXACTLY ONE required)
├── image1_descriptive_name.png (0-10 images allowed)
├── image2_another_name.jpg
└── ...
```

### Requirements

1. **Exactly ONE text file** (`.txt` or `.md`)
   - Contains the main memory content
   - This is what gets indexed in FAISS
   - Can have any filename

2. **Between 0-10 images** (optional)
   - Supported formats: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`
   - Must have descriptive filenames
   - Images should be mentioned/referenced in the text
   - Not indexed in FAISS (accessed on-demand)

3. **Descriptive folder name**
   - The folder name becomes the memory name
   - Use underscores or hyphens for multi-word names
   - Example: `vector_databases`, `setup-instructions`, `first_encounter`

## How It Works

### Text Search (Default)
When you query memories using RAG tools:
- Only the text content is searched via FAISS
- Results include metadata about available images
- Fast and efficient for text-based queries

### Image Access (On-Demand)
When you need visual information:
1. Query finds relevant memory (text-based)
2. Check metadata for available images
3. Use image tools to:
   - `list_memory_images` - See what images are available
   - `retrieve_memory_image` - Get image info
   - `analyze_memory_image` - Use GPT-4o vision to analyze the image in context

## Examples

### Example 1: Semantic Memory with Images

```
semantic/neural_networks/
├── definition.txt
├── architecture_diagram.png
├── activation_functions.png
└── training_process.jpg
```

**definition.txt:**
```
# Neural Networks

A neural network is a computational model inspired by biological neurons.

Key components (see architecture_diagram.png):
- Input layer
- Hidden layers
- Output layer

Common activation functions (see activation_functions.png):
- ReLU, Sigmoid, Tanh

The training process (see training_process.jpg) involves:
1. Forward propagation
2. Loss calculation
3. Backpropagation
4. Weight updates
```

### Example 2: Procedural Memory with Screenshots

```
procedural/docker_setup/
├── instructions.md
├── step1_installation.png
├── step2_verification.png
└── step3_first_container.png
```

**instructions.md:**
```
# Docker Setup Guide

## Step 1: Installation
Download Docker Desktop from docker.com (see step1_installation.png)

## Step 2: Verify Installation
Run `docker --version` to confirm (see step2_verification.png)

## Step 3: Run First Container
Execute `docker run hello-world` (see step3_first_container.png)
```

### Example 3: Episodic Memory with Photos

```
episodic/environment_exploration_2025_11_11/
├── session_notes.txt
├── initial_state.png
├── obstacle_encountered.jpg
└── final_position.png
```

**session_notes.txt:**
```
# Environment Exploration - November 11, 2025

Started in room A (initial_state.png). 
Encountered locked door (obstacle_encountered.jpg).
Found key and reached room B (final_position.png).

Lessons learned:
- Always check for keys before trying doors
- Map the environment systematically
```

## Using Images with the Agent

### Workflow

1. **Query text first:**
   ```
   User: What do you know about neural networks?
   Agent: [Uses query_semantic_memory, gets text + metadata about images]
   ```

2. **Agent sees image metadata in results:**
   ```
   Result includes:
   - memory_name: "neural_networks"
   - has_images: true
   - image_names: ["architecture_diagram.png", "activation_functions.png"]
   ```

3. **Agent can retrieve images if relevant:**
   ```
   Agent: [Uses list_memory_images to see all images]
   Agent: [Uses analyze_memory_image to understand visual content]
   ```

### Example Conversation

**User:** "Show me what a neural network architecture looks like"

**Agent Process:**
1. `query_semantic_memory("neural network architecture")` → Finds "neural_networks" memory
2. Sees metadata indicates images exist
3. `list_memory_images("semantic", "neural_networks")` → Lists available images
4. `analyze_memory_image("semantic", "neural_networks", "architecture_diagram.png", "Explain the neural network architecture shown")` → Analyzes image with GPT-4o vision
5. Provides comprehensive answer combining text and visual analysis

## Best Practices

### Text Content
- Reference images by name in the text
- Explain what each image shows
- Keep text self-contained (readable without images)

### Image Selection
- Include only relevant images (max 10)
- Use descriptive filenames
- Prefer diagrams and screenshots over photos
- Keep file sizes reasonable

### Organization
- One concept/procedure/episode per folder
- Group related memories in subdirectories if needed
- Use consistent naming conventions

## Migration from Old Format

If you have old single-file memories:
```bash
# Create folder
mkdir -p semantic/my_concept

# Move text into folder
mv semantic/my_concept.md semantic/my_concept/description.md

# Add images
cp relevant_images/* semantic/my_concept/
```

## Validation

The system automatically validates memory folders:
- ✅ Checks for exactly one text file
- ✅ Checks image count (0-10)
- ✅ Warns about invalid folders
- ✅ Skips invalid folders during indexing

Invalid folders are logged but don't break the system.

