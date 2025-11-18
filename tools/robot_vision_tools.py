"""
Robot vision analysis tools for the Experience Engine agent.
Provides vision analysis capabilities by directly capturing from the robot camera.
"""

import base64
import sys
import os
from typing import List, Optional
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Add utils to path and import Car
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.car import Car

# Global LLM instance (shared with the agent)
_llm: Optional[ChatOpenAI] = None

# Global car instance (shared with robot_tools)
from tools.robot_tools import get_car_instance


def initialize_robot_vision_tools(llm: ChatOpenAI) -> None:
    """
    Initialize robot vision tools with the agent's LLM.
    
    Args:
        llm: The ChatOpenAI LLM instance from the agent
    """
    global _llm
    _llm = llm


@tool
def analyze_robot_vision(prompt: str = "Describe what the robot sees in detail.") -> str:
    """
    Capture an image from the robot camera and analyze what it sees.
    This tool directly captures from the robot camera and analyzes the view in one step.
    
    Args:
        prompt: Prompt to guide the vision analysis (default: "Describe what the robot sees in detail.")
        
    Returns:
        String containing the vision analysis of what the robot sees
    """
    global _llm
    
    try:
        # Check if LLM is initialized
        if _llm is None:
            return "Error: Robot vision tools not initialized. Please initialize with an LLM."
        
        # Get car instance and capture image directly
        car = get_car_instance()
        img_bytes = car.capture_image()
        
        # Convert bytes to base64 directly (no filesystem operations)
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        # Create message with image using the shared LLM
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        )
        
        # Get response using the agent's LLM
        response = _llm.invoke([message])
        
        return f"Robot Vision Analysis:\n\n{response.content}"
        
    except Exception as e:
        return f"Error analyzing robot vision: {str(e)}"


@tool
def check_robot_vision_for(target: str) -> str:
    """
    Capture an image and check if the robot can see a specific object or feature.
    
    Args:
        target: What to look for (e.g., "a dog", "green circle", "obstacles", "doorway")
        
    Returns:
        String describing whether the target is visible and where it is located
    """
    prompt = f"Look carefully at this image and determine if you can see {target}. If yes, describe where it is located in the image (e.g., left, right, center, top, bottom) and provide any relevant details. If no, clearly state that {target} is not visible."
    return analyze_robot_vision.func(prompt)


@tool
def describe_robot_surroundings(dummy_input: str = "") -> str:
    """
    Capture an image and get a comprehensive description of the robot's surroundings.
    Analyzes objects, obstacles, colors, spatial layout, and navigation opportunities.
    
    Args:
        dummy_input: Unused parameter (for LangChain Tool compatibility)
        
    Returns:
        String containing detailed description of the robot's surroundings
    """
    prompt = """Analyze this robot camera view and provide a comprehensive description including:
1. Main objects and features visible
2. Obstacles or hazards the robot should be aware of
3. Color patterns and visual markers
4. Spatial layout (what's on the left, right, center, far, near)
5. Potential navigation paths or directions
6. Any interesting or notable details

Be specific and practical for robot navigation and exploration."""
    return analyze_robot_vision.func(prompt)


@tool
def detect_navigation_obstacles(dummy_input: str = "") -> str:
    """
    Capture an image and detect obstacles, assess navigation safety in the robot's view.
    
    Args:
        dummy_input: Unused parameter (for LangChain Tool compatibility)
        
    Returns:
        String describing obstacles and navigation recommendations
    """
    prompt = """Analyze this robot camera view specifically for navigation:
1. Identify any obstacles (walls, furniture, objects on the ground, etc.)
2. Assess which directions appear safe to move (forward, left, right, backward)
3. Estimate approximate distances to obstacles if possible
4. Identify any hazards (edges, drops, fragile objects, etc.)
5. Recommend the safest direction for movement

Provide practical navigation advice for a small mobile robot."""
    return analyze_robot_vision.func(prompt)


@tool
def identify_visual_targets(target_type: str = "any notable objects") -> str:
    """
    Capture an image and identify specific types of visual targets in the robot's view.
    Useful for searching or tracking specific objects.
    
    Args:
        target_type: Type of targets to look for (e.g., "colored markers", "animals", "people", "doors")
        
    Returns:
        String listing identified targets with their locations
    """
    prompt = f"""Look at this robot camera view and identify all instances of: {target_type}

For each target found:
1. Describe what it is
2. Specify its location in the frame (left/right/center, top/bottom)
3. Estimate relative distance (near/far)
4. Note any distinguishing features

If no targets of this type are found, clearly state that."""
    return analyze_robot_vision.func(prompt)


def create_robot_vision_tools() -> List:
    """
    Get all robot vision analysis tools as a list for use with LangChain AgentExecutor.
    
    Returns:
        List of robot vision analysis tools
    """
    return [
        analyze_robot_vision,
        check_robot_vision_for,
        describe_robot_surroundings,
        detect_navigation_obstacles,
        identify_visual_targets,
    ]

