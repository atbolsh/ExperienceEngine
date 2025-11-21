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

# Add utils to path and import Car
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.car import Car

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
        working_dir = os.path.join(os.path.dirname(__file__), '..', 'working')
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
        mode: "default" for 2-second turn with auto-stop, or "continuous" to turn until manually stopped.
        
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
        mode: "default" for 2-second turn with auto-stop, or "continuous" to turn until manually stopped.
        
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
        mode: "default" for 2-second movement with auto-stop, or "continuous" to move until manually stopped.
        
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
        mode: "default" for 2-second movement with auto-stop, or "continuous" to move until manually stopped.
        
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

