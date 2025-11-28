"""
Image injection utilities for attaching robot camera images to user inputs.
"""

import os
import sys
import base64
import cv2
from typing import Any, List, Optional, Union, Dict
from langchain.schema import HumanMessage

# Import from same environment directory
from .robot_tools import get_latest_robot_image, get_car_instance


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


def inject_image(user_input: str) -> Union[str, List[Dict[str, Any]]]:
    """
    Inject the latest robot camera image into the user input.
    
    This function takes a string input and returns either:
    - The original string if no image is available
    - A multi-modal list with text and image if an image is available
    
    Args:
        user_input: The user's text input
        
    Returns:
        Either the original string or a multi-modal content list with image
    """
    # Try to get the latest robot image
    image_data = None
    
    try:
        robot_image = get_latest_robot_image()
        
        if robot_image is not None:
            base64_data, image_format = encode_numpy_image_to_base64(robot_image)
            image_data = (base64_data, image_format, "latest robot camera capture")
    except Exception as e:
        # If robot image retrieval fails, try fallback
        pass
    
    # Fallback: Directly capture a new image from the robot camera
    if image_data is None:
        try:
            car = get_car_instance()
            img_bytes = car.capture_image()
            img_array = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
            
            if img_array is not None:
                base64_data, image_format = encode_numpy_image_to_base64(img_array)
                image_data = (base64_data, image_format, "fresh robot camera capture")
        except Exception as e:
            # If direct capture fails, no image will be included
            pass
    
    # If no image available, return original input
    if image_data is None:
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
