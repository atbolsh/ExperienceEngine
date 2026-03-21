# Robot Car Integration and Working Memory Setup

## Summary of Changes

This document outlines the recent changes to integrate robot car control and add working memory functionality.

## 1. Model stack

- **Text agent**: Local **Qwen3** small model (`llm_utils.get_local_llm`).
- **Vision / image analysis**: Local **Qwen2-VL** (`llm_utils.get_vision_llm`), used by environment and memory image tools—not the text agent.

## 2. Robot Car Tools

Created new robot interaction tools in `tools/robot_tools.py`:

### Camera System
- `capture_robot_image`: Captures an image from the robot's camera and saves it to `working/latest_capture.jpg`
- Images are automatically saved to working memory for reference

### Movement Controls
- `move_robot_forward`: Move forward (default 10cm, or "continuous")
- `move_robot_backward`: Move backward (default 10cm, or "continuous")
- `turn_robot_left`: Turn left (default 15°, or "continuous")
- `turn_robot_right`: Turn right (default 15°, or "continuous")
- `stop_robot_motion`: Emergency stop for all motion

### Key Features
- Discrete movements: Specify exact distance/angle (e.g., "15" for 15cm or 15°)
- Continuous motion: Pass "continuous" to move/turn until stopped
- All movement commands use the existing `utils/car.py` interface (not modified per user request)
- Graceful error handling if robot is not connected

## 3. Working Memory System

Added a new temporary memory category that is automatically cleared at session end:

### Vector Store Updates (`vector_store.py`)
- Added `working_store` to the vector store manager
- Added `working_path` and `working_index_path` for FAISS indices
- Implemented `clear_working_memory()` method to clear all working memory content
- Updated all store operations to include working memory

### Memory Tools Updates
- Created `tools/working_tools.py` with `query_working_memory` tool
- Added `write_working_memory` tool to `tools/memory_writing_tools.py`
- Updated `tools/memory_tools.py` to include working memory in refresh operations
- Added `clear_working_memory_function` for cleanup

### Session Cleanup (`main.py`)
- Modified `run_closing_sequence()` to automatically clear working memory at session end
- Cleanup happens after the closing reflection but before program exit
- Applies to all exit paths: user quit, agent-initiated end, agent restart, and Ctrl+C interrupt

## 4. Global Prompt Updates

Updated `prompts/global_prompt.txt` to include:

1. **Working Memory Documentation**
   - Explained that working memory is temporary and cleared at session end
   - Guidance on when to use working vs. long-term memory
   - Best practices for storing camera captures and observations

2. **Robot Car Interface Documentation**
   - Overview of all robot control capabilities
   - Camera system usage guidelines
   - Movement and turning controls
   - Safety features (emergency stop)

## 5. Tool Integration

Updated `tools/__init__.py` to:
- Import and register robot tools
- Import and register working memory tools
- Export `clear_working_memory_function` for session cleanup

## Directory Structure

```
experience-engine/
├── utils/
│   └── car.py                    # Robot car interface (NOT MODIFIED)
├── tools/
│   ├── robot_tools.py            # NEW: Robot control tools
│   ├── working_tools.py          # NEW: Working memory query tools
│   ├── memory_writing_tools.py   # UPDATED: Added write_working_memory
│   ├── memory_tools.py           # UPDATED: Added working memory support
│   └── __init__.py               # UPDATED: Register new tools
├── working/                      # Temporary memory directory (cleared on exit)
│   └── latest_capture.jpg        # Most recent robot camera capture
├── agent.py                      # Local Qwen text + vision via tools / llm_utils
├── main.py                       # UPDATED: Clear working memory on exit
├── vector_store.py               # UPDATED: Added working memory support
└── prompts/
    └── global_prompt.txt         # UPDATED: Added robot and working memory docs
```

## Usage Guidelines

### For Robot Exploration
1. Use `capture_robot_image` regularly to observe surroundings
2. Store observations in working memory during exploration
3. Move important findings to long-term memory (semantic/procedural/episodic)
4. Use precise movements for careful navigation
5. Use continuous movements for rapid exploration (don't forget to stop!)

### For Working Memory
- **DO USE** for: Recent captures, temporary notes, session-specific observations
- **DON'T USE** for: Important facts (semantic), procedures (procedural), or significant events (episodic)
- Remember: Working memory is wiped clean at the end of every session

### Image Storage
- Latest robot capture is always at `working/latest_capture.jpg`
- Images can be saved to any memory folder (semantic/procedural/episodic/working)
- Use image tools to analyze stored images from memories

## Testing Recommendations

Before deploying with the physical robot:

1. Test with robot disconnected (tools will handle gracefully)
2. Verify working memory creation and cleanup
3. Test continuous motion with stop commands
4. Verify image capture saves to working directory
5. Confirm working memory is cleared on all exit paths

## Notes

- The `utils/car.py` interface was not modified as requested
- Robot car instance is created lazily on first tool use
- Connection errors are non-fatal (tools report errors but continue)
- Working memory cleanup is automatic and requires no agent intervention
- Text and vision use separate local models (see `llm_utils.py`)

