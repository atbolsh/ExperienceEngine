"""
Memory folder schema definition.

Each memory is stored as a folder containing:
- Exactly one text file (.txt or .md)
- Between 0-10 images with descriptive names

This schema defines the structure and validation for memory folders.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MemoryFolder:
    """
    Represents a memory folder with its text content and images.
    
    Structure:
        memory_name/
        ├── description.txt (or any .txt/.md file - exactly one required)
        ├── image1_descriptive_name.png (0-10 images allowed)
        ├── image2_another_name.jpg
        └── ...
    """
    folder_name: str
    folder_path: Path
    text_file: Path
    text_content: str
    image_files: List[Path]
    
    @property
    def memory_name(self) -> str:
        """Get the memory name (folder name)."""
        return self.folder_name
    
    @property
    def image_names(self) -> List[str]:
        """Get list of image filenames."""
        return [img.name for img in self.image_files]
    
    def get_image_path(self, image_name: str) -> Optional[Path]:
        """Get the full path to an image by name."""
        for img in self.image_files:
            if img.name == image_name:
                return img
        return None


# Valid image extensions
VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}

# Valid text extensions
VALID_TEXT_EXTENSIONS = {'.txt', '.md'}


def validate_memory_folder(folder_path: Path) -> Tuple[bool, str]:
    """
    Validate that a folder follows the memory schema.
    
    Args:
        folder_path: Path to the memory folder
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not folder_path.is_dir():
        return False, f"{folder_path} is not a directory"
    
    # Get all files in the folder
    files = list(folder_path.iterdir())
    
    # Filter text and image files
    text_files = [f for f in files if f.suffix.lower() in VALID_TEXT_EXTENSIONS and f.is_file()]
    image_files = [f for f in files if f.suffix.lower() in VALID_IMAGE_EXTENSIONS and f.is_file()]
    
    # Must have exactly one text file
    if len(text_files) == 0:
        return False, f"No text file found in {folder_path.name}"
    if len(text_files) > 1:
        return False, f"Multiple text files found in {folder_path.name} (expected exactly one)"
    
    # Can have 0-10 images
    if len(image_files) > 10:
        return False, f"Too many images in {folder_path.name} (found {len(image_files)}, max 10)"
    
    return True, ""


def load_memory_folder(folder_path: Path) -> Optional[MemoryFolder]:
    """
    Load a memory folder and return a MemoryFolder object.
    
    Args:
        folder_path: Path to the memory folder
        
    Returns:
        MemoryFolder object or None if invalid
    """
    is_valid, error_msg = validate_memory_folder(folder_path)
    if not is_valid:
        print(f"Warning: {error_msg}")
        return None
    
    # Get all files
    files = list(folder_path.iterdir())
    
    # Get text file
    text_files = [f for f in files if f.suffix.lower() in VALID_TEXT_EXTENSIONS and f.is_file()]
    text_file = text_files[0]
    
    # Get image files
    image_files = sorted([
        f for f in files 
        if f.suffix.lower() in VALID_IMAGE_EXTENSIONS and f.is_file()
    ])
    
    # Read text content
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except Exception as e:
        print(f"Error reading {text_file}: {e}")
        return None
    
    return MemoryFolder(
        folder_name=folder_path.name,
        folder_path=folder_path,
        text_file=text_file,
        text_content=text_content,
        image_files=image_files
    )


def list_memory_folders(directory: Path) -> List[MemoryFolder]:
    """
    List all valid memory folders in a directory.
    
    Args:
        directory: Path to search for memory folders
        
    Returns:
        List of MemoryFolder objects
    """
    if not directory.exists():
        return []
    
    memory_folders = []
    
    # Iterate through subdirectories
    for item in directory.iterdir():
        if item.is_dir():
            memory = load_memory_folder(item)
            if memory:
                memory_folders.append(memory)
    
    return memory_folders


def get_memory_folder(directory: Path, memory_name: str) -> Optional[MemoryFolder]:
    """
    Get a specific memory folder by name.
    
    Args:
        directory: Parent directory containing memory folders
        memory_name: Name of the memory folder
        
    Returns:
        MemoryFolder object or None if not found
    """
    folder_path = directory / memory_name
    if not folder_path.exists():
        return None
    
    return load_memory_folder(folder_path)

