"""
Image injection utilities for attaching game screenshots to user inputs.

Note: With local text-only models like Qwen3 0.6B, images are NOT injected
into the main input stream. Instead, image analysis is done separately
via tools that call the local Qwen2-VL vision model (llm_utils.get_vision_llm).
"""

import os
import base64
import cv2
from typing import Any, List, Optional, Union, Dict
from langchain_compat import HumanMessage

# Import from same environment directory
from .game_tools import get_latest_game_image, get_game_instance


# Flag to control whether to inject images into main input
# Set to False when using local text-only models
USE_VISION_IN_MAIN_INPUT = False


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (base64_string, image_format)
    """
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Determine image format
    file_ext = os.path.splitext(image_path)[1].lower()
    image_format = file_ext[1:]  # Remove the dot
    if image_format == 'jpg':
        image_format = 'jpeg'
    
    return base64_data, image_format


def encode_numpy_image_to_base64(image_array) -> tuple[str, str]:
    """
    Encode a numpy image array to base64 string.
    
    Args:
        image_array: Numpy array representing an image
        
    Returns:
        Tuple of (base64_string, image_format)
    """
    # Encode the image as JPEG
    success, buffer = cv2.imencode('.jpg', image_array)
    if not success:
        raise ValueError("Failed to encode image array")
    
    base64_data = base64.b64encode(buffer).decode('utf-8')
    return base64_data, 'jpeg'


def inject_image(user_input: str) -> Union[str, HumanMessage]:
    """
    Inject the latest game screenshot into the user input.
    
    Note: When USE_VISION_IN_MAIN_INPUT is False (for local text models),
    this function simply returns the original text input but still updates
    the GUI viewer with the current game state.
    
    This function takes a string input and returns either:
    - The original string if no image is available or vision is disabled
    - A multi-modal message with text and image if vision is enabled and image available
    
    Args:
        user_input: The user's text input
        
    Returns:
        Either the original string or a multi-modal content list with image
    """
    # Try to get the latest game image
    image_data = None
    captured_image = None
    
    try:
        game_image = get_latest_game_image()
        
        if game_image is not None:
            captured_image = game_image
            if USE_VISION_IN_MAIN_INPUT:
                base64_data, image_format = encode_numpy_image_to_base64(game_image)
                image_data = (base64_data, image_format, "latest game capture")
    except Exception as e:
        # If game image retrieval fails, try fallback
        pass
    
    # Fallback: Directly capture a new image from the game
    if captured_image is None:
        try:
            game = get_game_instance()
            import numpy as np
            img_array = game.getData()
            
            # Scale to 0-255
            img_array = (img_array * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            if img_array is not None:
                captured_image = img_array
                if USE_VISION_IN_MAIN_INPUT:
                    base64_data, image_format = encode_numpy_image_to_base64(img_array)
                    image_data = (base64_data, image_format, "fresh game capture")
        except Exception as e:
            # If direct capture fails, no image will be included
            pass
    
    # Update GUI viewer with the captured image using blowup for crisp rendering
    if captured_image is not None:
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from gui_viewer import update_viewer
            
            # Use blowup method for crisp high-resolution rendering
            game = get_game_instance()
            
            # Calculate scale factor to make smaller dimension ~400 pixels
            h, w = captured_image.shape[:2]
            min_side = min(h, w)
            scale_factor = max(1.0, 400.0 / min_side)
            
            # Get high-resolution version using blowup
            import numpy as np
            hires_array = game.blowup(scale_factor)
            
            # Scale to 0-255
            hires_array = (hires_array * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            hires_array = cv2.cvtColor(hires_array, cv2.COLOR_RGB2BGR)
            
            update_viewer(hires_array)
        except Exception as e:
            pass  # Silently ignore GUI errors
    
    # If vision is disabled or no image available, return original input
    if not USE_VISION_IN_MAIN_INPUT or image_data is None:
        if captured_image is not None and not USE_VISION_IN_MAIN_INPUT:
            print("[Image] Game view updated in GUI (vision disabled for main input - use analyze_current_game_view tool)")
        return user_input
    
    base64_image, image_format, source_description = image_data
    
    # Create multi-modal content with image
    enhanced_text = user_input + " If applicable, please use the attached image to help respond to this query."
    
    multimodal_content = HumanMessage(content=[
        {"type": "text", "text": enhanced_text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{image_format};base64,{base64_image}"
            }
        }
    ])
    
    print(f"[Image Injection] Injecting {source_description} into user input")
    
    return multimodal_content

