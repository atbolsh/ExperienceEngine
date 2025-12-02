"""
Interactive prompt debugging environment for the game.
Import with: from prompt_test import *

Provides:
- reset_game_random(): Randomly reset the game
- reset_game_aligned(): Reset with agent facing the gold
- reset_game_unaligned(): Reset with agent NOT facing the gold
- analyze_current_game_view(): Analyze what the agent currently sees
"""

import os
import math
import random
from copy import deepcopy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up pygame to run in background with GUI
os.environ['SDL_VIDEODRIVER'] = 'x11'  # Use X11 for actual display

# Import game components
from environments.game_environment.discreteEngine import discreteGame

# Initialize GUI viewer
print("Initializing GUI viewer...")
try:
    from gui_viewer import initialize_viewer, load_gui_config, update_viewer
    import cv2
    import numpy as np
    
    # Enable GUI
    initialize_viewer(enabled=True, window_name="Game Environment - Debug Mode")
    _gui_available = True
    print("GUI viewer initialized successfully.")
except Exception as e:
    print(f"Warning: GUI viewer not available: {e}")
    _gui_available = False

# Create global game instance
print("Initializing game environment...")
_game = None

def _init_game():
    """Initialize the game instance if not already created."""
    global _game
    if _game is None:
        # Create initial settings with gameSize=224
        temp_game = discreteGame(envMode=True)
        initial_settings = temp_game.random_bare_settings(gameSize=224, max_agent_offset=0.8)
        _game = discreteGame(settings=initial_settings, envMode=True)
        _update_gui()
        print("Game environment created with gameSize=224.")
    return _game

def _update_gui():
    """Update the GUI viewer with the current game state."""
    if not _gui_available:
        return
    
    try:
        global _game
        if _game is None:
            return
            
        # Get current surface data (includes any drawings like arrows)
        img_array = _game.getData()
        
        # Scale to 0-255
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Calculate scale factor for ~400px min dimension
        scale_factor = max(1.0, 400.0 / _game.settings.gameSize)
        
        # Scale up the actual surface using cv2.resize (preserves arrow!)
        # Note: This is only for debugging. Main loop uses 'blowup' for better quality.
        if scale_factor > 1.0:
            new_width = int(img_array.shape[1] * scale_factor)
            new_height = int(img_array.shape[0] * scale_factor)
            img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        
        update_viewer(img_array)
    except Exception as e:
        print(f"Warning: Could not update GUI: {e}")

def reset_game_random():
    """
    Reset the game with random settings.
    Creates a simple level with agent + 1 gold + side walls only.
    
    Returns:
        Status message about the reset
    """
    global _game
    _game = _init_game()
    
    # Create a bare game with random positions and direction
    new_settings = _game.random_bare_settings(gameSize=224, max_agent_offset=0.8)
    
    # Apply the new settings
    _game.initial = deepcopy(new_settings)
    _game.settings = new_settings
    _game.reward = 0
    _game.universal_update()
    _update_gui()
    
    return "Game randomly reset. Call analyze_current_game_view() to see the current state."

def reset_game_aligned():
    """
    Reset the game with the agent facing directly at the gold.
    Uses random_bare_settings to create a simple level with 1 gold piece.
    
    Returns:
        Status message about the reset
    """
    global _game
    _game = _init_game()
    
    # Create a bare game (agent + 1 gold + side walls only)
    new_settings = _game.random_bare_settings(gameSize=224, max_agent_offset=0.8)
    
    # Get agent and gold positions
    agent_x = new_settings.agent_x
    agent_y = new_settings.agent_y
    gold_x, gold_y = new_settings.gold[0]
    
    # Calculate direction from agent to gold
    aligned_direction = _game.direction_angle(agent_x, agent_y, gold_x, gold_y)
    
    # Set the agent's direction to face the gold
    new_settings.direction = aligned_direction
    
    # Apply the new settings
    _game.initial = deepcopy(new_settings)
    _game.settings = new_settings
    _game.reward = 0
    _game.universal_update()
    _update_gui()
    
    return "Game reset with agent ALIGNED to face the gold. Call analyze_current_game_view() to verify."

def reset_game_unaligned():
    """
    Reset the game with the agent NOT facing the gold.
    Uses random_bare_settings and adds a perpendicular offset to the direction.
    
    Returns:
        Status message about the reset
    """
    global _game
    _game = _init_game()
    
    # Create a bare game (agent + 1 gold + side walls only)
    new_settings = _game.random_bare_settings(gameSize=224, max_agent_offset=0.8)
    
    # Get agent and gold positions
    agent_x = new_settings.agent_x
    agent_y = new_settings.agent_y
    gold_x, gold_y = new_settings.gold[0]
    
    # Calculate direction from agent to gold
    aligned_direction = _game.direction_angle(agent_x, agent_y, gold_x, gold_y)
    
    # Add a perpendicular offset (90 degrees or more)
    # Randomly choose between +90 to +180 degrees or -90 to -180 degrees
    offset_options = [
        random.uniform(math.pi/2, math.pi),      # 90-180 degrees
        random.uniform(-math.pi, -math.pi/2)     # -180 to -90 degrees
    ]
    offset = random.choice(offset_options)
    
    unaligned_direction = _game.mod2pi(aligned_direction + offset)
    
    # Set the agent's direction to NOT face the gold
    new_settings.direction = unaligned_direction
    
    # Apply the new settings
    _game.initial = deepcopy(new_settings)
    _game.settings = new_settings
    _game.reward = 0
    _game.universal_update()
    _update_gui()
    
    return "Game reset with agent UNALIGNED (not facing the gold). Call analyze_current_game_view() to verify."

def draw_arrow_to_gold():
    """
    Draw an arrow from the agent pointing towards the gold.
    This is a visual aid that shows the direction the agent would need to face
    to be aligned with the gold.
    
    Returns:
        Status message with the direction angle in radians
    """
    global _game
    if _game is None:
        _init_game()
    
    # Draw the arrow using the game's built-in method
    # Note: We don't call universal_update() because it redraws everything and erases the arrow
    pointing_angle = _game.bare_draw_arrow_at_gold()
    _update_gui()
    
    return f"Arrow drawn towards gold. Direction angle: {pointing_angle:.4f} radians ({math.degrees(pointing_angle):.2f} degrees)"

def analyze_current_game_view(query=""):
    """
    Analyze the current game view using vision AI.
    Preserves any drawings (like arrows) that are on the current surface.
    
    Args:
        query: Optional specific question about the game state
        
    Returns:
        Detailed analysis of what the agent currently sees
    """
    global _game
    if _game is None:
        _init_game()
    
    try:
        import base64
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        from datetime import datetime
        
        # Get current surface data (preserves any arrows or drawings)
        img_array = _game.getData()
        
        # Scale to 0-255
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Save to working memory
        working_dir = os.path.join(os.path.dirname(__file__), 'working')
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
            analysis_prompt = "Analyze this image from the game environment. Describe what you see, including: the agent (green circle with red eye), gold pieces (yellow/gold colored), walls (black), any arrows or lines drawn, and the spatial layout. Be specific and detailed about positions and orientations."
        
        # Create multi-modal message
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

# Initialize the game on import
print("\n" + "="*60)
print("PROMPT TEST ENVIRONMENT READY")
print("="*60)
print("\nAvailable functions:")
print("  - reset_game_random()      : Reset with random settings")
print("  - reset_game_aligned()     : Reset with agent facing gold")
print("  - reset_game_unaligned()   : Reset with agent NOT facing gold")
print("  - draw_arrow_to_gold()     : Draw arrow pointing to gold")
print("  - analyze_current_game_view(query='') : Analyze current view")
print("\nExample usage:")
print("  >>> reset_game_aligned()")
print("  >>> draw_arrow_to_gold()")
print("  >>> analyze_current_game_view('Is the agent facing the gold?')")
print("="*60 + "\n")

# Initialize game on import
_init_game()
print("Game initialized and ready for testing!\n")

