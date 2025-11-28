# Session Continuity Update

## Overview

This update adds automatic session continuity features to help the agent avoid repeating mistakes across sessions. The agent now automatically recalls context from previous sessions when it starts up.

## Changes Made

### 1. Context Prompt System (`prompts/context_prompt.md`)

**Created**: A new file `prompts/context_prompt.md` that stores learned hints about the environment.

**Initial Content**:
- "A red bag can look like any red splotch, not necessarily a stereotypical bag shape"
- "Avoid long chains of actions unless you are going through a simple loop (like 'rotate until you see X')"

**Constraints**:
- Maximum 1000 characters OR 10 lines (hard limit enforced)
- Agent must balance keeping valuable old information with adding new insights

### 2. New Context Editing Tools (`tools/context_tools.py`)

**Created**: Two new tools for managing context hints:

1. **`edit_context_prompt`**: Allows the agent to update context_prompt.md
   - Enforces 1000 char / 10 line limit
   - Validates content before saving
   - Shows old content when updating for transparency

2. **`read_context_prompt`**: Allows the agent to read current context hints
   - Displays current character and line count
   - Shows full content

### 3. Automatic Memory Loading (`agent.py`)

**Modified**: Agent initialization now automatically loads:

1. **Most Recent Episodic Memory**: 
   - Finds the most recently modified folder in `episodic/`
   - Loads its text content
   - Notes if it contains images
   - Provides continuity from the last session

2. **Context Prompt**:
   - Loads `prompts/context_prompt.md` if it exists
   - Shows learned hints at agent startup

3. **System Prompt Enhancement**:
   - Both are appended to the system prompt automatically
   - Clearly labeled sections for easy reference
   - Agent sees this information on every startup

### 4. Tool Registration (`tools/__init__.py`)

**Modified**: Registered the new context tools so they're available to the agent.

## How It Works

### At Startup

When the agent initializes:

1. Loads global_prompt.txt (existing behavior)
2. Loads environment_blurb (existing behavior)
3. **NEW**: Loads context_prompt.md with learned hints
4. **NEW**: Loads the most recent episodic memory
5. Appends both to the system prompt in a "continuity section"

### During Operation

The agent can:
- Read current context hints with `read_context_prompt`
- Update context hints with `edit_context_prompt` (respecting limits)
- Reference the most recent episodic memory already in its system prompt

### Example System Prompt Section

```
**LEARNED CONTEXT (from context_prompt.md):**
# Context Hints

1. **Red Bag Detection**: A red bag can look like any red splotch...
2. **Action Chains**: Avoid long chains unless...

**SESSION CONTINUITY:**
To help you understand what happened in the previous session, here is the most recent episodic memory:

=== MOST RECENT EPISODIC MEMORY ===
Memory: vision_test_red_bag_2025_11_26

[Content of the most recent memory...]

(This memory contains 2 images: test1.jpg, test2.jpg)
=== END OF RECENT MEMORY ===
```

## Benefits

1. **Reduced Repetition**: Agent automatically knows what happened last session
2. **Learned Hints**: Critical insights persist across sessions
3. **Automatic**: No manual loading required - happens on startup
4. **Concise**: Hard limits prevent prompt bloat (max 10 lines for context)
5. **Transparent**: Agent can read/edit context hints as needed

## Files Created

- `prompts/context_prompt.md` - Stores learned hints
- `tools/context_tools.py` - Tools for reading/editing context hints
- `SESSION_CONTINUITY_UPDATE.md` - This documentation

## Files Modified

- `agent.py` - Added functions to load context and recent memory; integrated into system prompt
- `tools/__init__.py` - Registered new context tools

## Usage

### For the Agent

The agent can now:
```python
# Read current context hints
read_context_prompt()

# Update context hints (respecting limits)
edit_context_prompt(new_content="# Updated hints\n1. New insight...")
```

### For Users

No action needed! The system automatically:
- Loads the most recent episodic memory on startup
- Shows learned context hints
- Makes both available in every conversation

## Future Enhancements

Possible improvements:
- Load multiple recent episodic memories (configurable count)
- Add timestamp tracking to context hints
- Allow agent to mark hints as "temporary" vs "permanent"
- Support different context files for different environments

