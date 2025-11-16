"""
Game environment interaction tools.
Provides wrappers for controlling the game agent and capturing images.
"""

import os
import base64
from io import BytesIO
from typing import List
import numpy as np
from PIL import Image

from langchain.tools import Tool


# Global game instance
_game_instance = None


def initialize_game(game_instance):
    """Initialize the global game instance."""
    global _game_instance
    _game_instance = game_instance


def get_game_instance():
    """Get the current game instance."""
    global _game_instance
    if _game_instance is None:
        # Initialize with default settings if not already initialized
        from game.discreteEngine import discreteGame
        from game.levels.skeleton import Settings
        _game_instance = discreteGame(envMode=True)
    return _game_instance


def numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded string."""
    # Normalize to 0-255 if needed
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    if len(image_array.shape) == 3:
        # Transpose from (width, height, channels) to (height, width, channels) if needed
        if image_array.shape[2] == 3:
            image_array = np.transpose(image_array, (1, 0, 2))
        image = Image.fromarray(image_array.astype(np.uint8), mode='RGB')
    else:
        image = Image.fromarray(image_array.astype(np.uint8))
    
    # Convert to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def save_game_image(image_array: np.ndarray, filepath: str):
    """Save a numpy array image to a file."""
    # Normalize to 0-255 if needed
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    if len(image_array.shape) == 3:
        # Transpose from (width, height, channels) to (height, width, channels) if needed
        if image_array.shape[2] == 3:
            image_array = np.transpose(image_array, (1, 0, 2))
        image = Image.fromarray(image_array.astype(np.uint8), mode='RGB')
    else:
        image = Image.fromarray(image_array.astype(np.uint8))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Save the image
    image.save(filepath)


def capture_image_tool(input_str: str = "") -> str:
    """
    Capture the current game state as an image.
    
    This captures a snapshot of the current playing field. The image shows:
    - The agent (green circle with red eye indicating direction)
    - Gold pieces (small gold circles)
    - Walls (black rectangles)
    
    Args:
        input_str: Optional filepath to save the image. If provided, saves to working memory.
                  Format: "path/to/image.png" or just "image_name" (will be saved in working/)
    
    Returns:
        Description of the captured image and where it was saved (if filepath provided).
    """
    game = get_game_instance()
    image_array = game.getData()
    
    result_msg = "Captured game state image. "
    
    if input_str.strip():
        # Save to specified location
        filepath = input_str.strip()
        
        # If no directory specified, save to working memory
        if '/' not in filepath:
            filepath = os.path.join("working", filepath)
        
        # Ensure .png extension
        if not filepath.endswith('.png'):
            filepath += '.png'
        
        save_game_image(image_array, filepath)
        result_msg += f"Saved to: {filepath}"
    else:
        result_msg += "Image captured but not saved. Provide a filepath to save it."
    
    # Return description
    result_msg += f"\n\nImage shows the game state with agent position, direction, walls, and gold pieces."
    result_msg += f"\nImage dimensions: {image_array.shape}"
    
    return result_msg


def swivel_clockwise_tool(input_str: str = "") -> str:
    """
    Rotate the agent clockwise.
    
    Rotates the agent's direction by π/30 radians (6 degrees) clockwise.
    This changes which direction the agent is facing.
    
    Returns:
        Confirmation message and any gold collected during the update.
    """
    game = get_game_instance()
    gold_collected = game.swivel_clock()
    
    msg = "Agent rotated clockwise (6 degrees)."
    if gold_collected > 0:
        msg += f" Collected {gold_collected} gold!"
    
    return msg


def swivel_counterclockwise_tool(input_str: str = "") -> str:
    """
    Rotate the agent counterclockwise (anticlockwise).
    
    Rotates the agent's direction by π/30 radians (6 degrees) counterclockwise.
    This changes which direction the agent is facing.
    
    Returns:
        Confirmation message and any gold collected during the update.
    """
    game = get_game_instance()
    gold_collected = game.swivel_anticlock()
    
    msg = "Agent rotated counterclockwise (6 degrees)."
    if gold_collected > 0:
        msg += f" Collected {gold_collected} gold!"
    
    return msg


def step_forward_tool(input_str: str = "") -> str:
    """
    Move the agent forward in the direction it's facing.
    
    Moves the agent forward by up to 1/16th of the game space (default limit).
    The agent will stop if it hits a wall. The agent automatically collects
    any gold it touches.
    
    Returns:
        Confirmation message and any gold collected.
    """
    game = get_game_instance()
    gold_collected = game.stepForward()
    
    msg = "Agent stepped forward."
    if gold_collected > 0:
        msg += f" Collected {gold_collected} gold!"
    
    return msg


def step_backward_tool(input_str: str = "") -> str:
    """
    Move the agent backward (opposite to the direction it's facing).
    
    Moves the agent backward by up to 1/16th of the game space (default limit).
    The agent will stop if it hits a wall. The agent automatically collects
    any gold it touches.
    
    Returns:
        Confirmation message and any gold collected.
    """
    game = get_game_instance()
    gold_collected = game.stepBackward()
    
    msg = "Agent stepped backward."
    if gold_collected > 0:
        msg += f" Collected {gold_collected} gold!"
    
    return msg


def create_game_tools() -> List[Tool]:
    """
    Create game environment interaction tools.
    
    Returns:
        List of game control tools
    """
    return [
        Tool(
            name="capture_game_image",
            func=capture_image_tool,
            description=(
                "Capture the current game state as an image. Shows the agent (green circle with red eye), "
                "gold pieces (small gold dots), and walls (black rectangles). "
                "Optionally provide a filename to save the image to working memory, e.g., 'current_state.png'. "
                "Use this frequently to observe the environment before and after actions."
            )
        ),
        Tool(
            name="swivel_clockwise",
            func=swivel_clockwise_tool,
            description=(
                "Rotate the agent clockwise by 6 degrees. Use this to change the direction the agent is facing. "
                "The red eye on the green circle shows which direction the agent faces."
            )
        ),
        Tool(
            name="swivel_counterclockwise",
            func=swivel_counterclockwise_tool,
            description=(
                "Rotate the agent counterclockwise (anticlockwise) by 6 degrees. Use this to change the direction "
                "the agent is facing. The red eye on the green circle shows which direction the agent faces."
            )
        ),
        Tool(
            name="step_forward",
            func=step_forward_tool,
            description=(
                "Move the agent forward in the direction it's currently facing. The agent will move up to 1/16th "
                "of the game space but will stop if it hits a wall. Automatically collects gold on contact."
            )
        ),
        Tool(
            name="step_backward",
            func=step_backward_tool,
            description=(
                "Move the agent backward (opposite to its facing direction). The agent will move up to 1/16th "
                "of the game space but will stop if it hits a wall. Automatically collects gold on contact."
            )
        ),
    ]

