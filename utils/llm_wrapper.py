"""
Custom LLM wrapper that automatically includes images with every message.
Injects the latest robot camera image or working memory image into all LLM calls.
"""

import os
import sys
import base64
import cv2
from typing import Any, List, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

# Add tools to path for robot imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.robot_tools import get_latest_robot_image, get_car_instance


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


class ImageInjectingLLM(ChatOpenAI):
    """
    Custom LLM wrapper that automatically injects images into all messages.
    
    Images are sourced from:
    1. Latest robot camera capture (via get_latest_robot_image())
    2. Fallback to most recent image in working memory
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_image_source: Optional[str] = None
    
    def _get_image_to_inject(self) -> Optional[tuple[str, str, str]]:
        """
        Get the image to inject into the message.
        
        Returns:
            Optional tuple of (base64_image, image_format, source_description)
        """
        # Try to get the latest robot image first
        try:
            robot_image = get_latest_robot_image()
            
            if robot_image is not None:
                base64_data, image_format = encode_numpy_image_to_base64(robot_image)
                return base64_data, image_format, "latest robot camera capture"
        except Exception as e:
            # If robot image retrieval fails, continue to fallback
            pass
        
        # Fallback: Directly capture a new image from the robot camera
        try:
            car = get_car_instance()
            img_bytes = car.capture_image()
            img_array = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
            
            if img_array is not None:
                base64_data, image_format = encode_numpy_image_to_base64(img_array)
                return base64_data, image_format, "fresh robot camera capture"
        except Exception as e:
            # If direct capture fails, no image will be included
            pass
        
        return None
    
    def _inject_image_into_messages(
        self, 
        messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """
        Inject an image into the last human message.
        
        Args:
            messages: List of messages
            
        Returns:
            Modified list of messages with image injected
        """
        image_data = self._get_image_to_inject()
        
        if image_data is None:
            # No image available, return messages unchanged
            return messages
        
        base64_image, image_format, source_description = image_data
        
        # Find the last human message and inject the image
        modified_messages = []
        image_injected = False
        
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            
            if isinstance(msg, HumanMessage) and not image_injected:
                # Convert text content to multi-modal content
                if isinstance(msg.content, str):
                    # Simple string content - convert to multi-modal
                    new_content = [
                        {"type": "text", "text": msg.content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        }
                    ]
                    modified_msg = HumanMessage(content=new_content)
                    modified_messages.insert(0, modified_msg)
                elif isinstance(msg.content, list):
                    # Already multi-modal - check if image already present
                    has_image = any(
                        isinstance(item, dict) and item.get("type") == "image_url"
                        for item in msg.content
                    )
                    
                    if not has_image:
                        # Add image to existing multi-modal content
                        new_content = msg.content + [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{base64_image}"
                                }
                            }
                        ]
                        modified_msg = HumanMessage(content=new_content)
                        modified_messages.insert(0, modified_msg)
                    else:
                        # Image already present, don't duplicate
                        modified_messages.insert(0, msg)
                else:
                    # Unknown content type, keep as is
                    modified_messages.insert(0, msg)
                
                image_injected = True
            else:
                modified_messages.insert(0, msg)
        
        return modified_messages
    
    def invoke(
        self,
        input: Union[str, List[BaseMessage]],
        *args,
        **kwargs
    ) -> Any:
        """Override invoke to inject images."""
        # Convert input to messages if it's a string
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input
        
        # Inject image into messages
        modified_messages = self._inject_image_into_messages(messages)
        
        # Call parent invoke with modified messages
        return super().invoke(modified_messages, *args, **kwargs)
    
    async def ainvoke(
        self,
        input: Union[str, List[BaseMessage]],
        *args,
        **kwargs
    ) -> Any:
        """Override async invoke to inject images."""
        # Convert input to messages if it's a string
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input
        
        # Inject image into messages
        modified_messages = self._inject_image_into_messages(messages)
        
        # Call parent ainvoke with modified messages
        return super().ainvoke(modified_messages, *args, **kwargs)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        *args,
        **kwargs
    ) -> Any:
        """Override _generate to inject images."""
        # Inject image into messages
        modified_messages = self._inject_image_into_messages(messages)
        
        # Call parent _generate with modified messages
        return super()._generate(modified_messages, *args, **kwargs)
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        *args,
        **kwargs
    ) -> Any:
        """Override async _generate to inject images."""
        # Inject image into messages
        modified_messages = self._inject_image_into_messages(messages)
        
        # Call parent _agenerate with modified messages
        return super()._agenerate(modified_messages, *args, **kwargs)

