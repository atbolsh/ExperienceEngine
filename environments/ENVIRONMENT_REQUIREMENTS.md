# Environment Requirements

This document describes the structure and requirements for creating new environments in the Experience Engine agent system.

## Overview

Environments are modular packages that provide the agent with tools and capabilities to interact with specific physical or virtual contexts. Each environment should be self-contained and follow a consistent structure.

## Environment Structure

Each environment should be organized as follows:

```
environments/
└── [environment_name]/
    ├── __init__.py           # Package initialization and exports
    ├── [core_module].py      # Core functionality (e.g., car.py)
    ├── [tool_module].py      # Tool definitions (e.g., robot_tools.py)
    └── llm_wrapper.py        # Image/data injection for LLM
```

## Required Components

Every environment MUST include the following components:

### 1. Image Injection Function (`inject_image`)

**Purpose**: Prepares and injects sensory data (typically images) into the LLM's input stream.

**Requirements**:
- Function signature: `inject_image(user_input: str) -> Union[str, HumanMessage]`
- Should return either plain text or a multi-modal message with image data
- Should handle cases where no image is available gracefully

**Location**: Typically in `llm_wrapper.py`

**Example**:
```python
def inject_image(user_input: str) -> Union[str, List[Dict[str, Any]]]:
    """
    Inject the latest captured image into the user input.
    Returns either original string or multi-modal content with image.
    """
    # Implementation here
    pass
```

### 2. Image Viewing Tool

**Purpose**: Allows the agent to capture and view the current state of the environment.

**Requirements**:
- Tool name: `capture_[environment]_image` (e.g., `capture_robot_image`)
- Should capture image and make it available for subsequent LLM calls
- Should return a status message indicating success or failure
- Should save images to `working/` directory for reference

**Example**:
```python
def capture_robot_image(dummy_input: str = "") -> str:
    """
    Capture an image from the robot's camera.
    The captured image becomes available to the LLM for the next call.
    """
    # Implementation here
    pass
```

### 3. Image Detection Tool

**Purpose**: Captures and immediately saves an image as `working/latest_capture.jpg`.

**Requirements**:
- Should capture fresh image from environment
- Must save to `working/latest_capture.jpg`
- Should also save with unique timestamp for history
- Should return confirmation message

**Example**:
```python
def capture_robot_image(dummy_input: str = "") -> str:
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    img_filename = f'capture_{timestamp}.jpg'
    img_path = os.path.join(working_dir, img_filename)
    cv2.imwrite(img_path, image)
    
    # Also save as latest_capture.jpg
    latest_path = os.path.join(working_dir, 'latest_capture.jpg')
    cv2.imwrite(latest_path, image)
    
    return f"Successfully captured image..."
```

### 4. Environment Navigation/Interaction Tools

**Purpose**: Provide the agent with tools to navigate and interact with the environment.

**Requirements**:
- Tools should be self-descriptive with clear docstrings
- Each tool should return status messages
- Tools should handle errors gracefully
- Consider both discrete actions and continuous control modes

**Example Tools** (for car environment):
- `move_forward` / `move_backward`
- `turn_left` / `turn_right`
- `stop_motion`
- Any environment-specific actions

### 5. Environment Initialization and Cleanup

**Purpose**: Setup and teardown functions for the environment.

**Requirements**:
- `initialize_[environment]()` function to set up connections/resources
- `close_[environment]()` function to clean up resources
- Functions should be exported in `__init__.py`

**Example**:
```python
def initialize_car():
    """Initialize the global robot car instance."""
    global car_instance
    try:
        car_instance = Car()
        car_instance.start()
    except Exception as e:
        print(f"Warning: Could not connect: {e}")

def close_car():
    """Close the robot car connection."""
    if car_instance is not None:
        car_instance.close()
```

## Package Exports

The environment's `__init__.py` should export:
- All tool creation functions
- Initialization and cleanup functions
- The `inject_image` function
- Any core classes or utilities

**Example `__init__.py`**:
```python
from .robot_tools import (
    initialize_car,
    close_car,
    create_robot_tools,
    # ... other functions
)

from .llm_wrapper import inject_image

from .car import Car

__all__ = [
    'Car',
    'initialize_car',
    'close_car',
    'create_robot_tools',
    'inject_image',
    # ... other exports
]
```

## Integration with Main System

To integrate a new environment:

1. **Create the environment package** in `environments/[environment_name]/`

2. **Update `select_environment.config`** to specify the active environment:
   ```
   ACTIVE_ENVIRONMENT=new_environment
   ```

3. **Update `tools/__init__.py`** to import from the new environment:
   ```python
   from environments.new_environment import create_tools, initialize_env, close_env
   ```

4. **Update `main.py`** to import `inject_image` from the new environment:
   ```python
   from environments.new_environment import inject_image
   ```

5. **Update `agent.py`** to call the environment's initialization:
   ```python
   from tools import initialize_env
   # ... in create_agent():
   initialize_env()
   ```

## Best Practices

1. **Error Handling**: All tools should handle errors gracefully and return informative error messages rather than throwing exceptions.

2. **Resource Management**: Always implement proper cleanup in the close function.

3. **Image Persistence**: Save all captured images with unique timestamps to maintain history.

4. **Status Messages**: Tools should return clear, human-readable status messages that the LLM can understand.

5. **Documentation**: All tools should have clear docstrings explaining their purpose, parameters, and return values.

6. **Testing**: Create test scripts to verify environment functionality before integration.

## Reference Implementation

See `environments/car_environment/` for a complete reference implementation that demonstrates all required components.

## Future Considerations

When creating new environments, consider:
- What sensory inputs are available? (images, sensor data, etc.)
- What actions can the agent perform?
- How does the environment provide feedback?
- Are there safety constraints or limits?
- What initialization/cleanup is required?
- Can the environment be simulated for testing?

---

**Note**: This is a living document. Update it as new patterns or requirements emerge from implementing additional environments.

