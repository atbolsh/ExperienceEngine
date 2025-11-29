"""
Game interaction tools for controlling the agent in the discrete game environment.
Provides interface to the game's movement and camera systems.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional
from langchain.tools import Tool

# Global game instance
game_instance = None

# Latest captured image
_latest_image = None


def initialize_game(game = None):
    """Initialize the global game instance."""
    global game_instance
    try:
        if game is None:
            print("Initializing game environment...")
            os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Run pygame in headless mode
            from .init_game import create_default_game
            game_instance = create_default_game()
        else:
            game_instance = game
        print("Game environment initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize game: {e}")
        print("Game tools will be available but may not function properly.")


def close_game():
    """Close the game instance."""
    global game_instance
    if game_instance is not None:
        try:
            print("Closing game environment...")
            # The game doesn't need explicit cleanup in envMode
            game_instance = None
            print("Game environment closed.")
        except Exception as e:
            print(f"Error closing game: {str(e)}")


def get_game_instance():
    """Get the global game instance."""
    if game_instance is None:
        raise RuntimeError("Game not initialized. Please restart the application.")
    return game_instance


def capture_game_image(dummy_input: str = "") -> str:
    """
    Capture an image from the game's current state.
    The captured image becomes available to the LLM for the next call.
    
    Args:
        dummy_input: Unused parameter (for LangChain Tool compatibility)
        
    Returns:
        Status message indicating successful capture
    """
    global _latest_image
    try:
        game = get_game_instance()
        # Get the current game state as numpy array
        # getData returns array with shape (width, height, 3) and values in [0, 1]
        img_array = game.getData()
        
        # Convert from (width, height, 3) to (height, width, 3) and scale to 0-255
        img_array = np.transpose(img_array, (1, 0, 2))  # Swap width and height
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        _latest_image = img_array
        
        # Save to working memory for reference
        working_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'working')
        os.makedirs(working_dir, exist_ok=True)
        
        # Save with timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        img_filename = f'game_capture_{timestamp}.jpg'
        img_path = os.path.join(working_dir, img_filename)
        cv2.imwrite(img_path, img_array)
        
        # Also save as latest_capture.jpg for backward compatibility
        latest_path = os.path.join(working_dir, 'latest_capture.jpg')
        cv2.imwrite(latest_path, img_array)
        
        return f"Successfully captured image from game. Image saved to working/{img_filename} and working/latest_capture.jpg. The image is now available for analysis."
    except Exception as e:
        return f"Error capturing image: {str(e)}"


def move_game_forward(dummy_input: str = "") -> str:
    """
    Move the game agent forward.
    
    Args:
        dummy_input: Unused parameter (for LangChain Tool compatibility)
        
    Returns:
        Status message including any reward collected
    """
    try:
        game = get_game_instance()
        gold_collected = game.stepForward()
        
        if gold_collected > 0:
            return f"Moved forward. Collected {gold_collected} gold! Total reward: {game.reward}"
        else:
            return "Moved forward."
    except Exception as e:
        return f"Error moving forward: {str(e)}"


def move_game_backward(dummy_input: str = "") -> str:
    """
    Move the game agent backward.
    
    Args:
        dummy_input: Unused parameter (for LangChain Tool compatibility)
        
    Returns:
        Status message including any reward collected
    """
    try:
        game = get_game_instance()
        gold_collected = game.stepBackward()
        
        if gold_collected > 0:
            return f"Moved backward. Collected {gold_collected} gold! Total reward: {game.reward}"
        else:
            return "Moved backward."
    except Exception as e:
        return f"Error moving backward: {str(e)}"


def turn_game_clockwise(dummy_input: str = "") -> str:
    """
    Turn the game agent clockwise (to the right).
    
    Args:
        dummy_input: Unused parameter (for LangChain Tool compatibility)
        
    Returns:
        Status message
    """
    try:
        game = get_game_instance()
        game.swivel_clock()
        return "Turned clockwise."
    except Exception as e:
        return f"Error turning clockwise: {str(e)}"


def turn_game_counterclockwise(dummy_input: str = "") -> str:
    """
    Turn the game agent counterclockwise (to the left).
    
    Args:
        dummy_input: Unused parameter (for LangChain Tool compatibility)
        
    Returns:
        Status message
    """
    try:
        game = get_game_instance()
        game.swivel_anticlock()
        return "Turned counterclockwise."
    except Exception as e:
        return f"Error turning counterclockwise: {str(e)}"


def get_latest_game_image():
    """
    Get the latest captured image for LLM processing.
    This is called internally to attach images to LLM calls.
    
    Returns:
        The latest captured image as a numpy array, or None if no image captured yet.
    """
    return _latest_image


def analyze_current_game_view(query: str = "") -> str:
    """
    Capture a fresh image from the game and provide an instant analysis.
    This tool is particularly useful in the middle of a tool chain when you need
    to check the current game state without waiting for the next user interaction.
    
    Unlike capture_game_image which just captures the image for later analysis,
    this tool captures AND analyzes the image immediately, returning a detailed
    description of what the agent currently sees.
    
    Args:
        query: Optional specific question about the game state (e.g., "where is the gold?", 
               "what obstacles are visible?", "describe the agent's position")
        
    Returns:
        A detailed text description of the current game state, or error message
    """
    global _latest_image
    try:
        # Import here to avoid circular dependency
        import base64
        from langchain_openai import ChatOpenAI
        
        # Capture fresh image
        game = get_game_instance()
        img_array = game.getData()
        
        # Convert from (width, height, 3) to (height, width, 3) and scale to 0-255
        img_array = np.transpose(img_array, (1, 0, 2))
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        if img_array is None:
            return "Error: Failed to decode captured image."
        
        # Update the latest image
        _latest_image = img_array
        
        # Save to working memory
        working_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'working')
        os.makedirs(working_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_filename = f'game_analysis_{timestamp}.jpg'
        img_path = os.path.join(working_dir, img_filename)
        cv2.imwrite(img_path, img_array)
        
        # Encode image to base64 for vision model
        success, buffer = cv2.imencode('.jpg', img_array)
        if not success:
            return "Error: Failed to encode image for analysis."
        
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Create vision-enabled LLM
        llm = ChatOpenAI(
            model="gpt-5",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Prepare analysis prompt
        if query:
            analysis_prompt = f"Analyze this image from the game environment and answer the following: {query}\n\nProvide a clear, concise, and detailed response."
        else:
            analysis_prompt = "Analyze this image from the game environment. Describe what you see, including: the agent (green circle with red eye), gold pieces (yellow/gold colored), walls (black), and the spatial layout. Be specific and detailed about positions and orientations."
        
        # Create multi-modal message
        from langchain.schema import HumanMessage
        message = HumanMessage(content=[
            {"type": "text", "text": analysis_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ])
        
        # Get analysis from vision model
        response = llm.invoke([message])
        
        # Return the analysis with metadata
        result = f"[Game Analysis - Captured at {timestamp}]\n"
        result += f"Image saved to: working/{img_filename}\n\n"
        result += f"Analysis:\n{response.content}"
        
        return result
        
    except Exception as e:
        return f"Error analyzing current view: {str(e)}"


def create_game_tools() -> List[Tool]:
    """
    Create and return all game interaction tools.
    
    Returns:
        List of game control and camera tools
    """
    return [
        Tool(
            name="capture_game_image",
            func=capture_game_image,
            description="Capture an image from the game's current state. The image will be available for analysis and saved to working memory. Call this before analyzing the game environment."
        ),
        Tool(
            name="analyze_current_game_view",
            func=analyze_current_game_view,
            description="Capture and analyze the game's current state in one step. This is especially useful in the MIDDLE of a tool chain when you need to check what the agent sees right now, without waiting for the next user interaction. Returns a detailed text description of the current view. You can optionally provide a specific question about the game state (e.g., 'where is the gold?', 'what obstacles are visible?'). Use this when you need immediate visual feedback during a multi-step task."
        ),
        Tool(
            name="move_game_forward",
            func=move_game_forward,
            description="Move the game agent forward in the direction it's currently facing. Returns status and any gold collected."
        ),
        Tool(
            name="move_game_backward",
            func=move_game_backward,
            description="Move the game agent backward (opposite to the direction it's facing). Returns status and any gold collected."
        ),
        Tool(
            name="turn_game_clockwise",
            func=turn_game_clockwise,
            description="Turn the game agent clockwise (to the right). Changes the agent's facing direction."
        ),
        Tool(
            name="turn_game_counterclockwise",
            func=turn_game_counterclockwise,
            description="Turn the game agent counterclockwise (to the left). Changes the agent's facing direction."
        ),
    ]

