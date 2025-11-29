"""
Robot interaction tools for controlling the car and capturing images.
Provides interface to the physical robot's movement and camera systems.
"""

import os
import sys
import cv2
import time
from datetime import datetime
from typing import List, Optional
from langchain.tools import Tool

# Import Car from same directory
from .car import Car

# Global car instance
car_instance: Optional[Car] = None

# Latest captured image
_latest_image = None


def initialize_car():
    """Initialize the global robot car instance."""
    global car_instance
    try:
        print("Initializing robot car connection...")
        car_instance = Car()
        car_instance.start()
        print("Robot car connected successfully.")
    except Exception as e:
        print(f"Warning: Could not connect to robot car: {e}")
        print("Robot tools will be available but may not function properly.")


def close_car():
    """Close the robot car connection."""
    global car_instance
    if car_instance is not None:
        try:
            print("Closing robot car connection...")
            car_instance.close()
            print("Robot car connection closed.")
        except Exception as e:
            print(f"Error closing robot car: {str(e)}")


def get_car_instance():
    """Get the global car instance."""
    if car_instance is None:
        raise RuntimeError("Robot car not initialized. Please restart the application.")
    return car_instance


def capture_robot_image(dummy_input: str = "") -> str:
    """
    Capture an image from the robot's camera.
    The captured image becomes available to the LLM for the next call.
    
    Args:
        dummy_input: Unused parameter (for LangChain Tool compatibility)
        
    Returns:
        Status message indicating successful capture
    """
    global _latest_image
    try:
        car = get_car_instance()
        img_bytes = car.capture_image()
        _latest_image = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
        
        # Save to working memory for reference
        working_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'working')
        os.makedirs(working_dir, exist_ok=True)
        
        # Save with timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        img_filename = f'capture_{timestamp}.jpg'
        img_path = os.path.join(working_dir, img_filename)
        cv2.imwrite(img_path, _latest_image)
        
        # Also save as latest_capture.jpg for backward compatibility
        latest_path = os.path.join(working_dir, 'latest_capture.jpg')
        cv2.imwrite(latest_path, _latest_image)
        
        return f"Successfully captured image from robot camera. Image saved to working/{img_filename} and working/latest_capture.jpg. The image is now available for analysis."
    except Exception as e:
        return f"Error capturing image: {str(e)}"


def turn_robot_left(mode: str = "default") -> str:
    """
    Turn the robot left.
    
    Args:
        mode: "default" for 1-second turn with auto-stop, or "continuous" to turn until manually stopped.
        
    Returns:
        Status message
    """
    try:
        car = get_car_instance()
        
        if mode.lower() == "continuous":
            car.left(speed=40)
            return "Robot is now turning left continuously. Use stop_motion to halt."
        else:
            car.left(speed=40)
            time.sleep(1)
            car.stop()
            return "Robot turned left for 1 second."
    except Exception as e:
        return f"Error turning left: {str(e)}"


def turn_robot_right(mode: str = "default") -> str:
    """
    Turn the robot right.
    
    Args:
        mode: "default" for 1-second turn with auto-stop, or "continuous" to turn until manually stopped.
        
    Returns:
        Status message
    """
    try:
        car = get_car_instance()
        
        if mode.lower() == "continuous":
            car.right(speed=40)
            return "Robot is now turning right continuously. Use stop_motion to halt."
        else:
            car.right(speed=40)
            time.sleep(1)
            car.stop()
            return "Robot turned right for 1 second."
    except Exception as e:
        return f"Error turning right: {str(e)}"


def move_robot_forward(mode: str = "default") -> str:
    """
    Move the robot forward.
    
    Args:
        mode: "default" for 1-second movement with auto-stop, or "continuous" to move until manually stopped.
        
    Returns:
        Status message
    """
    try:
        car = get_car_instance()
        
        if mode.lower() == "continuous":
            car.forward(speed=40)
            return "Robot is now moving forward continuously. Use stop_motion to halt."
        else:
            car.forward(speed=40)
            time.sleep(1)
            car.stop()
            return "Robot moved forward for 1 second."
    except Exception as e:
        return f"Error moving forward: {str(e)}"


def move_robot_backward(mode: str = "default") -> str:
    """
    Move the robot backward.
    
    Args:
        mode: "default" for 1-second movement with auto-stop, or "continuous" to move until manually stopped.
        
    Returns:
        Status message
    """
    try:
        car = get_car_instance()
        
        if mode.lower() == "continuous":
            car.backward(speed=40)
            return "Robot is now moving backward continuously. Use stop_motion to halt."
        else:
            car.backward(speed=40)
            time.sleep(1)
            car.stop()
            return "Robot moved backward for 1 second."
    except Exception as e:
        return f"Error moving backward: {str(e)}"


def stop_robot_motion(dummy_input: str = "") -> str:
    """
    Stop all robot motion immediately.
    
    Args:
        dummy_input: Unused parameter (for LangChain Tool compatibility)
        
    Returns:
        Status message
    """
    try:
        car = get_car_instance()
        car.stop()
        return "Robot motion stopped."
    except Exception as e:
        return f"Error stopping robot: {str(e)}"


def get_latest_robot_image():
    """
    Get the latest captured image for LLM processing.
    This is called internally to attach images to LLM calls.
    
    Returns:
        The latest captured image as a numpy array, or None if no image captured yet.
    """
    return _latest_image


def analyze_current_view(query: str = "") -> str:
    """
    Capture a fresh image from the robot's camera and provide an instant analysis.
    This tool is particularly useful in the middle of a tool chain when you need
    to check the current environment without waiting for the next user interaction.
    
    Unlike capture_robot_image which just captures the image for later analysis,
    this tool captures AND analyzes the image immediately, returning a detailed
    description of what the robot currently sees.
    
    Args:
        query: Optional specific question about the environment (e.g., "what objects 
               are visible?", "is there anything blocking the path?", "describe colors")
        
    Returns:
        A detailed text description of the current robot camera view, or error message
    """
    global _latest_image
    try:
        # Import here to avoid circular dependency
        import base64
        from langchain_openai import ChatOpenAI
        
        # Capture fresh image
        car = get_car_instance()
        img_bytes = car.capture_image()
        img_array = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
        
        if img_array is None:
            return "Error: Failed to decode captured image."
        
        # Update the latest image
        _latest_image = img_array
        
        # Save to working memory
        working_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'working')
        os.makedirs(working_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_filename = f'analysis_{timestamp}.jpg'
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
            analysis_prompt = f"Analyze this image from the robot's camera and answer the following: {query}\n\nProvide a clear, concise, and detailed response."
        else:
            analysis_prompt = "Analyze this image from the robot's camera. Describe what you see, including: objects, colors, spatial layout, any notable features, and the general environment. Be specific and detailed."
        
        # Create multi-modal message
        from langchain.schema import HumanMessage
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
        result = f"[Image Analysis - Captured at {timestamp}]\n"
        result += f"Image saved to: working/{img_filename}\n\n"
        result += f"Analysis:\n{response.content}"
        
        return result
        
    except Exception as e:
        return f"Error analyzing current view: {str(e)}"


def create_robot_tools() -> List[Tool]:
    """
    Create and return all robot interaction tools.
    
    Returns:
        List of robot control and camera tools
    """
    return [
        Tool(
            name="capture_robot_image",
            func=capture_robot_image,
            description="Capture an image from the robot's camera. The image will be available for analysis and saved to working memory. Call this before analyzing the robot's surroundings."
        ),
        Tool(
            name="analyze_current_view",
            func=analyze_current_view,
            description="Capture and analyze the robot's current camera view in one step. This is especially useful in the MIDDLE of a tool chain when you need to check what the robot sees right now, without waiting for the next user interaction. Returns a detailed text description of the current view. You can optionally provide a specific question about the environment (e.g., 'what objects are visible?', 'is there anything blocking the path?'). Use this when you need immediate visual feedback during a multi-step task."
        ),
        Tool(
            name="turn_robot_left",
            func=turn_robot_left,
            description="Turn the robot left for 1 second at speed 40, then auto-stop. Pass 'continuous' to turn continuously until manually stopped. Example: 'default' or 'continuous'"
        ),
        Tool(
            name="turn_robot_right",
            func=turn_robot_right,
            description="Turn the robot right for 1 second at speed 40, then auto-stop. Pass 'continuous' to turn continuously until manually stopped. Example: 'default' or 'continuous'"
        ),
        Tool(
            name="move_robot_forward",
            func=move_robot_forward,
            description="Move the robot forward for 1 second at speed 40, then auto-stop. Pass 'continuous' to move continuously until manually stopped. Example: 'default' or 'continuous'"
        ),
        Tool(
            name="move_robot_backward",
            func=move_robot_backward,
            description="Move the robot backward for 1 second at speed 40, then auto-stop. Pass 'continuous' to move continuously until manually stopped. Example: 'default' or 'continuous'"
        ),
        Tool(
            name="stop_robot_motion",
            func=stop_robot_motion,
            description="Stop all robot motion immediately. Use this to halt continuous movement or turning."
        ),
    ]

