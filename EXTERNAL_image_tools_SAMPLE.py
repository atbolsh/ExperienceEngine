"""
Image analysis tools for the RAG agent.
Provides image analysis capabilities using the agent's main LLM (GPT-5).
"""

import os
import base64
from typing import List, Optional
from langchain_compat import HumanMessage, tool
from langchain_openai import ChatOpenAI


# Global LLM instance (shared with the agent)
_llm: Optional[ChatOpenAI] = None


def initialize_image_tools(llm: ChatOpenAI) -> None:
    """
    Initialize image tools with the agent's LLM.
    
    Args:
        llm: The ChatOpenAI LLM instance from the agent
    """
    global _llm
    _llm = llm


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@tool
def analyze_image(image_path: str, prompt: str = "Describe this image in detail.") -> str:
    """
    Analyze an image and provide detailed information about its contents.
    
    Args:
        image_path: Path to the image file (supports jpg, jpeg, png, gif, webp)
        prompt: Optional prompt to guide the analysis (default: "Describe this image in detail.")
        
    Returns:
        String containing the image analysis
    """
    global _llm
    
    try:
        # Check if LLM is initialized
        if _llm is None:
            return "Error: Image tools not initialized. Please initialize with an LLM."
        
        # Check if file exists
        if not os.path.exists(image_path):
            return f"Error: Image file '{image_path}' does not exist"
        
        # Check if it's a file
        if not os.path.isfile(image_path):
            return f"Error: '{image_path}' is not a file"
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in valid_extensions:
            return f"Error: Unsupported image format '{file_ext}'. Supported formats: {', '.join(valid_extensions)}"
        
        # Encode image to base64
        base64_image = encode_image(image_path)
        
        # Determine image format for data URL
        image_format = file_ext[1:]  # Remove the dot
        if image_format == 'jpg':
            image_format = 'jpeg'
        
        # Create message with image using the shared LLM
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{base64_image}"
                    }
                }
            ]
        )
        
        # Get response using the agent's LLM
        response = _llm.invoke([message])
        
        return f"Image Analysis for '{image_path}':\n\n{response.content}"
        
    except Exception as e:
        return f"Error analyzing image '{image_path}': {str(e)}"


@tool
def analyze_image_with_question(image_path: str, question: str) -> str:
    """
    Analyze an image and answer a specific question about it.
    
    Args:
        image_path: Path to the image file
        question: Specific question to answer about the image
        
    Returns:
        String containing the answer to the question
    """
    return analyze_image.func(image_path, question)


@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR capabilities.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        String containing extracted text from the image
    """
    prompt = "Extract all text visible in this image. Provide the text exactly as it appears, maintaining the original formatting and structure as much as possible."
    return analyze_image.func(image_path, prompt)


@tool
def compare_images(image_path1: str, image_path2: str) -> str:
    """
    Compare two images and describe their similarities and differences.
    Note: This function analyzes images sequentially and provides a comparison based on their descriptions.
    
    Args:
        image_path1: Path to the first image file
        image_path2: Path to the second image file
        
    Returns:
        String containing the comparison analysis
    """
    try:
        # Analyze first image
        result1 = analyze_image.func(image_path1, "Describe this image in detail, focusing on key objects, colors, composition, and notable features.")
        if "Error" in result1:
            return result1
        
        # Analyze second image
        result2 = analyze_image.func(image_path2, "Describe this image in detail, focusing on key objects, colors, composition, and notable features.")
        if "Error" in result2:
            return result2
        
        # Create comparison summary
        comparison = f"Image Comparison:\n\n"
        comparison += f"First Image ({image_path1}):\n{result1}\n\n"
        comparison += f"Second Image ({image_path2}):\n{result2}\n\n"
        comparison += "Note: This is a sequential analysis. For detailed comparison, review both descriptions above."
        
        return comparison
        
    except Exception as e:
        return f"Error comparing images: {str(e)}"


def get_image_tools() -> List:
    """
    Get all image analysis tools as a list for use with LangChain AgentExecutor.
    
    Returns:
        List of image analysis tools
    """
    return [
        analyze_image,
        analyze_image_with_question,
        extract_text_from_image,
        compare_images
    ]

