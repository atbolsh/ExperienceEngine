"""
Utility functions for image handling, including unique filename generation.
"""

import os
import cv2
from datetime import datetime
from typing import Optional


def generate_unique_image_filename(prefix: str = "image", extension: str = "jpg") -> str:
    """
    Generate a unique image filename with timestamp.
    
    Args:
        prefix: Prefix for the filename (default: "image")
        extension: File extension without dot (default: "jpg")
        
    Returns:
        Unique filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    return f"{prefix}_{timestamp}.{extension}"


def save_image_with_unique_name(
    image_array, 
    directory: str, 
    prefix: str = "image",
    extension: str = "jpg",
    also_save_as_latest: bool = False
) -> str:
    """
    Save an image with a unique timestamp-based filename.
    
    Args:
        image_array: NumPy array representing the image
        directory: Directory to save the image
        prefix: Prefix for the filename (default: "image")
        extension: File extension without dot (default: "jpg")
        also_save_as_latest: If True, also save as "latest_{prefix}.{extension}"
        
    Returns:
        Path to the saved image file
    """
    os.makedirs(directory, exist_ok=True)
    
    # Generate unique filename
    filename = generate_unique_image_filename(prefix, extension)
    filepath = os.path.join(directory, filename)
    
    # Save the image
    cv2.imwrite(filepath, image_array)
    
    # Optionally save as latest
    if also_save_as_latest:
        latest_filename = f"latest_{prefix}.{extension}"
        latest_path = os.path.join(directory, latest_filename)
        cv2.imwrite(latest_path, image_array)
    
    return filepath


def get_unique_filepath(directory: str, filename: str) -> str:
    """
    Get a unique filepath by adding timestamp if file exists.
    
    Args:
        directory: Directory for the file
        filename: Desired filename
        
    Returns:
        Unique filepath (may have timestamp added)
    """
    filepath = os.path.join(directory, filename)
    
    # If file doesn't exist, return as is
    if not os.path.exists(filepath):
        return filepath
    
    # File exists - add timestamp to make it unique
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    unique_filename = f"{name}_{timestamp}{ext}"
    
    return os.path.join(directory, unique_filename)

