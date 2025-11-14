"""
Image retrieval and analysis tools for memories.
Allows the agent to access images from memory folders when relevant.
"""

import base64
from pathlib import Path
from typing import List, Optional

from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import os

from tools.memory_schema import get_memory_folder


# Memory directories
SEMANTIC_DIR = Path("semantic")
PROCEDURAL_DIR = Path("procedural")
EPISODIC_DIR = Path("episodic")


# ===== Input Schemas =====

class ListMemoryImagesInput(BaseModel):
    """Input for list_memory_images tool."""
    memory_type: str = Field(description="Type of memory: 'semantic', 'procedural', or 'episodic'")
    memory_name: str = Field(description="Name of the memory folder")


class RetrieveMemoryImageInput(BaseModel):
    """Input for retrieve_memory_image tool."""
    memory_type: str = Field(description="Type of memory: 'semantic', 'procedural', or 'episodic'")
    memory_name: str = Field(description="Name of the memory folder")
    image_name: str = Field(description="Name of the image file")


class AnalyzeMemoryImageInput(BaseModel):
    """Input for analyze_memory_image tool."""
    memory_type: str = Field(description="Type of memory: 'semantic', 'procedural', or 'episodic'")
    memory_name: str = Field(description="Name of the memory folder")
    image_name: str = Field(description="Name of the image file")
    question: str = Field(description="Question to ask about the image in context of the memory")


# ===== Helper Functions =====

def get_memory_directory(memory_type: str) -> Optional[Path]:
    """Get the directory for a memory type."""
    if memory_type == "semantic":
        return SEMANTIC_DIR
    elif memory_type == "procedural":
        return PROCEDURAL_DIR
    elif memory_type == "episodic":
        return EPISODIC_DIR
    else:
        return None


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ===== Tool Functions =====

def list_memory_images(memory_type: str, memory_name: str) -> str:
    """
    List all images available in a specific memory folder.
    
    Args:
        memory_type: Type of memory ('semantic', 'procedural', 'episodic')
        memory_name: Name of the memory folder
        
    Returns:
        List of image names or error message
    """
    try:
        directory = get_memory_directory(memory_type)
        if not directory:
            return f"Error: Invalid memory type '{memory_type}'. Must be 'semantic', 'procedural', or 'episodic'."
        
        memory = get_memory_folder(directory, memory_name)
        if not memory:
            return f"Error: Memory '{memory_name}' not found in {memory_type} directory."
        
        if len(memory.image_files) == 0:
            return f"Memory '{memory_name}' has no images."
        
        output = [f"Images in {memory_type} memory '{memory_name}':"]
        for i, img in enumerate(memory.image_files, 1):
            size = img.stat().st_size
            output.append(f"  {i}. {img.name} ({size} bytes)")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Error listing images: {str(e)}"


def retrieve_memory_image(memory_type: str, memory_name: str, image_name: str) -> str:
    """
    Retrieve information about a specific image from a memory folder.
    Returns the image path and basic information.
    
    Args:
        memory_type: Type of memory ('semantic', 'procedural', 'episodic')
        memory_name: Name of the memory folder
        image_name: Name of the image file
        
    Returns:
        Image information or error message
    """
    try:
        directory = get_memory_directory(memory_type)
        if not directory:
            return f"Error: Invalid memory type '{memory_type}'."
        
        memory = get_memory_folder(directory, memory_name)
        if not memory:
            return f"Error: Memory '{memory_name}' not found."
        
        image_path = memory.get_image_path(image_name)
        if not image_path:
            available = ", ".join(memory.image_names) if memory.image_names else "none"
            return f"Error: Image '{image_name}' not found in memory '{memory_name}'. Available images: {available}"
        
        size = image_path.stat().st_size
        
        return f"Image '{image_name}' found in {memory_type} memory '{memory_name}'.\nPath: {image_path}\nSize: {size} bytes\nExtension: {image_path.suffix}"
    
    except Exception as e:
        return f"Error retrieving image: {str(e)}"


def analyze_memory_image(memory_type: str, memory_name: str, image_name: str, question: str) -> str:
    """
    Analyze an image from a memory folder in the context of the memory's text and a specific question.
    Uses GPT-4o vision capabilities to analyze the image.
    
    Args:
        memory_type: Type of memory ('semantic', 'procedural', 'episodic')
        memory_name: Name of the memory folder
        image_name: Name of the image file
        question: Question to ask about the image
        
    Returns:
        Analysis result or error message
    """
    try:
        directory = get_memory_directory(memory_type)
        if not directory:
            return f"Error: Invalid memory type '{memory_type}'."
        
        memory = get_memory_folder(directory, memory_name)
        if not memory:
            return f"Error: Memory '{memory_name}' not found."
        
        image_path = memory.get_image_path(image_name)
        if not image_path:
            return f"Error: Image '{image_name}' not found in memory '{memory_name}'."
        
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)
        image_ext = image_path.suffix.lower()
        
        # Determine MIME type
        mime_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        mime_type = mime_type_map.get(image_ext, 'image/png')
        
        # Create vision-enabled LLM
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create prompt with memory context
        prompt = f"""You are analyzing an image from a memory folder.

Memory Type: {memory_type}
Memory Name: {memory_name}
Image Name: {image_name}

Memory Text Context:
{memory.text_content}

Question: {question}

Please analyze the image in the context of the memory text and answer the question."""
        
        # Call vision API
        from langchain_core.messages import HumanMessage
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    }
                }
            ]
        )
        
        response = llm.invoke([message])
        
        return f"Analysis of '{image_name}' from {memory_type} memory '{memory_name}':\n\n{response.content}"
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


# ===== Tool Definitions =====

def create_image_tools() -> List[Tool]:
    """Create and return image retrieval and analysis tools."""
    return [
        StructuredTool(
            name="list_memory_images",
            func=list_memory_images,
            description="List all images available in a specific memory folder. Use when you want to see what images are associated with a memory. Inputs: memory_type (string: 'semantic', 'procedural', or 'episodic'), memory_name (string: name of the memory folder)",
            args_schema=ListMemoryImagesInput,
        ),
        StructuredTool(
            name="retrieve_memory_image",
            func=retrieve_memory_image,
            description="Get information about a specific image from a memory folder (path, size, etc). Inputs: memory_type, memory_name, image_name",
            args_schema=RetrieveMemoryImageInput,
        ),
        StructuredTool(
            name="analyze_memory_image",
            func=analyze_memory_image,
            description="Analyze an image from a memory folder using vision AI in the context of the memory's text. Use when you need to understand visual content to answer a question. Inputs: memory_type, memory_name, image_name, question (what you want to know about the image)",
            args_schema=AnalyzeMemoryImageInput,
        ),
    ]

