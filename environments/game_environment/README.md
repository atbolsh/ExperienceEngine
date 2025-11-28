# Game Environment

This environment provides the agent with tools and capabilities to interact with a discrete 2D game world.

## Overview

The game environment enables the agent to:
- Capture images from the game's current state
- Control the agent's movement (forward, backward, turn clockwise, turn counterclockwise)
- Analyze the game's visual state in real-time
- Inject game screenshots into LLM conversations

## Components

### `discreteEngine.py`
Core game engine that simulates a 2D world with:
- Agent (green circle with red eye indicator)
- Gold pieces (yellow/gold colored objects to collect)
- Walls (black obstacles)
- Physics and collision detection

### `levels/skeleton.py`
Game settings and level configuration system

### `game_tools.py`
LangChain tools that wrap the game functionality for use by the agent:
- `capture_game_image` - Capture and save an image from the game's current state
- `analyze_current_game_view` - Capture and immediately analyze the current game state
- `move_game_forward` - Move the agent forward
- `move_game_backward` - Move the agent backward
- `turn_game_clockwise` - Rotate the agent clockwise (right)
- `turn_game_counterclockwise` - Rotate the agent counterclockwise (left)
- `create_game_tools()` - Factory function to create all game tools

### `llm_wrapper.py`
Image injection utilities that attach game screenshots to LLM inputs:
- `inject_image(user_input)` - Main function to inject images into messages
- `encode_image_to_base64()` - Utility to encode images for vision models
- `encode_numpy_image_to_base64()` - Utility to encode numpy arrays as images

## Usage

### Initialization

```python
from environments.game_environment import initialize_game, create_game_tools

# Initialize the game
initialize_game()

# Create tools for the agent
tools = create_game_tools()
```

### Custom Game Initialization

You can also initialize with a custom game instance:

```python
from environments.game_environment.discreteEngine import discreteGame
from environments.game_environment.levels.skeleton import Settings
from environments.game_environment import initialize_game

# Create custom settings
settings = Settings(
    gameSize=64,
    agent_x=0.5,
    agent_y=0.5,
    agent_r=0.05,
    gold_r=0.015,
    gold=[[0.3, 0.3], [0.7, 0.7]],
    walls=[...]
)

game = discreteGame(settings=settings, envMode=True)
initialize_game(game)
```

### Image Injection

```python
from environments.game_environment import inject_image

# Inject latest game screenshot into user input
enhanced_input = inject_image("What do you see?")
```

### Cleanup

```python
from environments.game_environment import close_game

# Close the game environment
close_game()
```

## Configuration

The game environment can be configured with:
- `gameSize`: Size of the game window (default: 64x64 pixels)
- Agent properties: position, radius, direction
- Gold properties: positions, radius
- Walls: position, dimensions, rotation

## Image Storage

All captured images are saved to the `working/` directory:
- Timestamped files: `game_capture_YYYYMMDD_HHMMSS_mmm.jpg`
- Latest capture: `latest_capture.jpg`

## Game Mechanics

### Agent
- Represented as a green circle with a red "eye" showing direction
- Can move forward/backward and rotate
- Collects gold on contact

### Gold
- Yellow/gold colored circles
- Collected when agent touches them
- Adds to reward score

### Walls
- Black rectangular obstacles
- Can be rotated at any angle
- Block agent movement

### Movement
- **Forward**: Move in the direction the agent is facing
- **Backward**: Move opposite to the facing direction
- **Clockwise**: Rotate right by π/30 radians (~6 degrees)
- **Counterclockwise**: Rotate left by π/30 radians (~6 degrees)

## Headless Mode

The game runs in headless mode (no window) using pygame with SDL_VIDEODRIVER set to 'dummy'. This allows the agent to interact with the game programmatically without requiring a display.

## Dependencies

- `pygame` - Game engine and rendering
- `opencv-cv2` - Image processing
- `numpy` - Array handling
- `pillow` - Image manipulation
- `langchain` - Tool creation
- `langchain_openai` - Vision model for image analysis

## Future Enhancements

Potential improvements for this environment:
- Multiple difficulty levels
- Procedurally generated levels
- More complex obstacles and mechanics
- Path planning tools
- Multi-agent scenarios
- Time limits or score challenges

