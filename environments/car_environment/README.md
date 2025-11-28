# Car Environment

This environment provides the agent with tools and capabilities to interact with a robot car.

## Overview

The car environment enables the agent to:
- Capture images from the robot's camera
- Control the robot's movement (forward, backward, left, right)
- Analyze the robot's visual perspective in real-time
- Inject camera images into LLM conversations

## Components

### `car.py`
Core Car class that handles low-level communication with the robot hardware via socket connection. Provides methods for:
- Movement control (forward, backward, left, right, stop)
- Camera control (capture images, rotate camera)
- Sensor readings (distance, MPU, ground detection)

### `robot_tools.py`
LangChain tools that wrap the Car class functionality for use by the agent:
- `capture_robot_image` - Capture and save an image from the robot's camera
- `analyze_current_view` - Capture and immediately analyze the current view
- `turn_robot_left` / `turn_robot_right` - Rotate the robot
- `move_robot_forward` / `move_robot_backward` - Move the robot
- `stop_robot_motion` - Stop all robot movement
- `create_robot_tools()` - Factory function to create all robot tools

### `llm_wrapper.py`
Image injection utilities that attach robot camera images to LLM inputs:
- `inject_image(user_input)` - Main function to inject images into messages
- `encode_image_to_base64()` - Utility to encode images for vision models
- `encode_numpy_image_to_base64()` - Utility to encode numpy arrays as images

## Usage

### Initialization

```python
from environments.car_environment import initialize_car, create_robot_tools

# Initialize the car connection
initialize_car()

# Create tools for the agent
tools = create_robot_tools()
```

### Image Injection

```python
from environments.car_environment import inject_image

# Inject latest camera image into user input
enhanced_input = inject_image("What do you see?")
```

### Cleanup

```python
from environments.car_environment import close_car

# Close the car connection
close_car()
```

## Configuration

The robot car connects to IP `192.168.4.1` on port `100` by default. These settings are configured in `car.py`.

## Image Storage

All captured images are saved to the `working/` directory:
- Timestamped files: `capture_YYYYMMDD_HHMMSS_mmm.jpg`
- Latest capture: `latest_capture.jpg`

## Tool Modes

Movement tools support two modes:
- **default**: Execute action for 1 second then auto-stop
- **continuous**: Execute action until manually stopped

Example:
```python
# Move forward for 1 second (default)
move_robot_forward("default")

# Move forward continuously until stopped
move_robot_forward("continuous")
stop_robot_motion()
```

## Dependencies

- `opencv-cv2` - Image processing
- `numpy` - Array handling
- `langchain` - Tool creation
- `langchain_openai` - Vision model for image analysis

## Future Enhancements

Potential improvements for this environment:
- Configurable robot IP/port via config file
- More sophisticated camera controls
- Path planning tools
- Obstacle detection tools
- Battery/status monitoring

