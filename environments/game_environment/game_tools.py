"""
Game interaction tools for controlling the agent in the discrete game environment.
Provides interface to the game's movement and camera systems.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional
from langchain_compat import StructuredTool, Tool

# Global game instance
game_instance = None

# Latest captured image
_latest_image = None


def initialize_game(game = None):
    """Initialize the global game instance and GUI viewer."""
    global game_instance
    try:
        if game is None:
            print("Initializing game environment...")
            os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Run pygame in headless mode
            from .init_game import default_game
            game_instance = default_game()
        else:
            game_instance = game
        
        # Initialize GUI viewer
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from gui_viewer import (
                initialize_viewer,
                load_gui_config,
                get_display_mode,
                visual_updates_enabled,
            )
            gui_config_on = load_gui_config()
            if visual_updates_enabled(gui_config_on):
                initialize_viewer(enabled=True, window_name="Game Environment")
                if get_display_mode() == "jupyter":
                    print("Environment view: Jupyter mode (OpenCV window disabled; use notebook image widget).")
                else:
                    print("GUI viewer enabled.")
                
                # Capture and display initial image
                capture_game_image()
                print("Initial game view captured and displayed.")
        except Exception as e:
            print(f"GUI viewer could not be initialized: {e}")
        
        print("Game environment initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize game: {e}")
        print("Game tools will be available but may not function properly.")


def close_game():
    """Close the game instance and GUI viewer."""
    global game_instance
    if game_instance is not None:
        try:
            print("Closing game environment...")
            # The game doesn't need explicit cleanup in envMode
            game_instance = None
            
            # Close GUI viewer
            try:
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
                from gui_viewer import close_viewer
                close_viewer()
            except Exception as e:
                pass  # Silently ignore if viewer was never initialized
            
            print("Game environment closed.")
        except Exception as e:
            print(f"Error closing game: {str(e)}")


def get_game_instance():
    """Get the global game instance."""
    if game_instance is None:
        raise RuntimeError("Game not initialized. Please restart the application.")
    return game_instance


def _update_gui_viewer():
    """Update the GUI viewer with the current game state (without saving files)."""
    try:
        game = get_game_instance()
        
        # Get the current game state as numpy array
        img_array = game.getData()
        
        # Scale to 0-255
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Update GUI viewer with high-resolution version
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from gui_viewer import update_viewer
        
        # Calculate scale factor to make smaller dimension ~400 pixels
        h, w = img_array.shape[:2]
        min_side = min(h, w)
        scale_factor = max(1.0, 400.0 / min_side)
        
        # Get high-resolution version using blowup
        hires_array = game.blowup(scale_factor)
        
        # Scale to 0-255
        hires_array = (hires_array * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        hires_array = cv2.cvtColor(hires_array, cv2.COLOR_RGB2BGR)
        
        update_viewer(hires_array)
    except Exception as e:
        pass  # Silently ignore GUI errors


def capture_game_image(*args, **kwargs) -> str:
    """
    Capture an image from the game's current state.
    The captured image becomes available to the LLM for the next call.
    
    Returns:
        Status message indicating successful capture
    """
    global _latest_image
    try:
        game = get_game_instance()
        # Get the current game state as numpy array
        # getData returns array with shape (width, height, 3) and values in [0, 1]
        img_array = game.getData()
        
        # Scale to 0-255
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        _latest_image = img_array
        
        # Update GUI viewer with high-resolution version for game
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from gui_viewer import update_viewer
            
            # Use blowup method for crisp high-resolution rendering
            # Calculate scale factor to make smaller dimension ~400 pixels
            h, w = img_array.shape[:2]
            min_side = min(h, w)
            scale_factor = max(1.0, 400.0 / min_side)
            
            # Get high-resolution version using blowup
            hires_array = game.blowup(scale_factor)
            
            # Scale to 0-255
            hires_array = (hires_array * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            hires_array = cv2.cvtColor(hires_array, cv2.COLOR_RGB2BGR)
            
            update_viewer(hires_array)
        except Exception as e:
            pass  # Silently ignore GUI errors
        
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


def move_game_forward(*args, **kwargs) -> str:
    """
    Move the game agent forward.
    
    Returns:
        Status message including any reward collected
    """
    try:
        game = get_game_instance()
        gold_collected = game.stepForward()
        _update_gui_viewer()
        
        if gold_collected > 0:
            return f"Moved forward. Collected {gold_collected} gold! Total reward: {game.reward}"
        else:
            return "Moved forward."
    except Exception as e:
        return f"Error moving forward: {str(e)}"


def move_game_backward(*args, **kwargs) -> str:
    """
    Move the game agent backward.
    
    Returns:
        Status message including any reward collected
    """
    try:
        game = get_game_instance()
        gold_collected = game.stepBackward()
        _update_gui_viewer()
        
        if gold_collected > 0:
            return f"Moved backward. Collected {gold_collected} gold! Total reward: {game.reward}"
        else:
            return "Moved backward."
    except Exception as e:
        return f"Error moving backward: {str(e)}"


def turn_game_clockwise(*args, **kwargs) -> str:
    """
    Turn the game agent clockwise (to the right).
    
    Returns:
        Status message
    """
    try:
        game = get_game_instance()
        game.swivel_clock()
        _update_gui_viewer()
        return "Turned clockwise."
    except Exception as e:
        return f"Error turning clockwise: {str(e)}"


def turn_game_counterclockwise(*args, **kwargs) -> str:
    """
    Turn the game agent counterclockwise (to the left).
    
    Returns:
        Status message
    """
    try:
        game = get_game_instance()
        game.swivel_anticlock()
        _update_gui_viewer()
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
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from llm_utils import get_vision_llm
        
        # Capture fresh image
        game = get_game_instance()
        img_array = game.getData()
        
        # Scale to 0-255
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        if img_array is None:
            return "Error: Failed to decode captured image."
        
        # Update the latest image
        _latest_image = img_array
        
        # Update GUI viewer with high-resolution version for game
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from gui_viewer import update_viewer
            
            # Use blowup method for crisp high-resolution rendering
            # Calculate scale factor to make smaller dimension ~400 pixels
            h, w = img_array.shape[:2]
            min_side = min(h, w)
            scale_factor = max(1.0, 400.0 / min_side)
            
            # Get high-resolution version using blowup
            hires_array = game.blowup(scale_factor)
            
            # Scale to 0-255
            hires_array = (hires_array * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            hires_array = cv2.cvtColor(hires_array, cv2.COLOR_RGB2BGR)
            
            update_viewer(hires_array)
        except Exception as e:
            pass  # Silently ignore GUI errors
            hires_array = np.transpose(hires_array, (1, 0, 2))
            hires_array = (hires_array * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            hires_array = cv2.cvtColor(hires_array, cv2.COLOR_RGB2BGR)
            
            update_viewer(hires_array)
        except Exception as e:
            pass  # Silently ignore GUI errors
        
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
        
        # Use GPT-5 vision model for image analysis
        llm = get_vision_llm()
        
        # Prepare analysis prompt
        if query:
            analysis_prompt = f"Analyze this image from the game environment and answer the following: {query}\n\nProvide a clear, concise, and detailed response."
        else:
            analysis_prompt = "Analyze this image from the game environment. Describe what you see, including: the agent (green circle with red eye), gold pieces (yellow/gold colored), walls (black), and the spatial layout. Be specific and detailed about positions and orientations."
        
        # Create multi-modal message
        from langchain_compat import HumanMessage
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
        StructuredTool.from_function(
            func=capture_game_image,
            name="capture_game_image",
            description="Capture an image from the game's current state. The image will be available for analysis and saved to working memory. Call this before analyzing the game environment.",
            args_schema=None
        ),
        Tool(
            name="analyze_current_game_view",
            func=analyze_current_game_view,
            description="Capture and analyze the game's current state in one step. This is especially useful in the MIDDLE of a tool chain when you need to check what the agent sees right now, without waiting for the next user interaction. Returns a detailed text description of the current view. You can optionally provide a specific question about the game state (e.g., 'where is the gold?', 'what obstacles are visible?'). Use this when you need immediate visual feedback during a multi-step task."
        ),
        StructuredTool.from_function(
            func=move_game_forward,
            name="move_game_forward",
            description="Move the game agent forward in the direction it's currently facing. Returns status and any gold collected.",
            args_schema=None
        ),
        StructuredTool.from_function(
            func=move_game_backward,
            name="move_game_backward",
            description="Move the game agent backward (opposite to the direction it's facing). Returns status and any gold collected.",
            args_schema=None
        ),
        StructuredTool.from_function(
            func=turn_game_clockwise,
            name="turn_game_clockwise",
            description="Turn the game agent clockwise (to the right). Changes the agent's facing direction.",
            args_schema=None
        ),
        StructuredTool.from_function(
            func=turn_game_counterclockwise,
            name="turn_game_counterclockwise",
            description="Turn the game agent counterclockwise (to the left). Changes the agent's facing direction.",
            args_schema=None
        ),
    ]

