"""
Robot interaction tools for controlling the car and capturing images.
Provides interface to the physical robot's movement and camera systems.
"""

import os
import cv2
from typing import List
from langchain.tools import Tool

# Latest captured image
_latest_image = None


def get_car_instance():
    """Get the global car instance from main module."""
    import main
    if main.car_instance is None:
        raise RuntimeError("Robot car not initialized. Please restart the application.")
    return main.car_instance


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
        
        # Save the latest image
        img_path = os.path.join(working_dir, 'latest_capture.jpg')
        cv2.imwrite(img_path, _latest_image)
        
        return f"Successfully captured image from robot camera. Image saved to working/latest_capture.jpg. The image is now available for analysis."
    except Exception as e:
        return f"Error capturing image: {str(e)}"


def turn_robot_left(angle_str: str = "15") -> str:
    """
    Turn the robot left by a specified angle.
    
    Args:
        angle_str: Angle in degrees to turn (default: 15). Use "continuous" to turn continuously until stopped.
        
    Returns:
        Status message
    """
    try:
        car = get_car_instance()
        
        if angle_str.lower() == "continuous":
            car.left()
            return "Robot is now turning left continuously. Use stop_motion to halt."
        else:
            angle = float(angle_str)
            car.left(angle=angle)
            return f"Robot turned left by {angle} degrees."
    except ValueError:
        return f"Invalid angle: {angle_str}. Please provide a number or 'continuous'."
    except Exception as e:
        return f"Error turning left: {str(e)}"


def turn_robot_right(angle_str: str = "15") -> str:
    """
    Turn the robot right by a specified angle.
    
    Args:
        angle_str: Angle in degrees to turn (default: 15). Use "continuous" to turn continuously until stopped.
        
    Returns:
        Status message
    """
    try:
        car = get_car_instance()
        
        if angle_str.lower() == "continuous":
            car.right()
            return "Robot is now turning right continuously. Use stop_motion to halt."
        else:
            angle = float(angle_str)
            car.right(angle=angle)
            return f"Robot turned right by {angle} degrees."
    except ValueError:
        return f"Invalid angle: {angle_str}. Please provide a number or 'continuous'."
    except Exception as e:
        return f"Error turning right: {str(e)}"


def move_robot_forward(distance_str: str = "10") -> str:
    """
    Move the robot forward by a specified distance.
    
    Args:
        distance_str: Distance in centimeters to move (default: 10). Use "continuous" to move continuously until stopped.
        
    Returns:
        Status message
    """
    try:
        car = get_car_instance()
        
        if distance_str.lower() == "continuous":
            car.forward()
            return "Robot is now moving forward continuously. Use stop_motion to halt."
        else:
            distance = float(distance_str)
            car.forward(distance=distance)
            return f"Robot moved forward {distance} cm."
    except ValueError:
        return f"Invalid distance: {distance_str}. Please provide a number or 'continuous'."
    except Exception as e:
        return f"Error moving forward: {str(e)}"


def move_robot_backward(distance_str: str = "10") -> str:
    """
    Move the robot backward by a specified distance.
    
    Args:
        distance_str: Distance in centimeters to move (default: 10). Use "continuous" to move continuously until stopped.
        
    Returns:
        Status message
    """
    try:
        car = get_car_instance()
        
        if distance_str.lower() == "continuous":
            car.backward()
            return "Robot is now moving backward continuously. Use stop_motion to halt."
        else:
            distance = float(distance_str)
            car.backward(distance=distance)
            return f"Robot moved backward {distance} cm."
    except ValueError:
        return f"Invalid distance: {distance_str}. Please provide a number or 'continuous'."
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
            description="Turn the robot left. Provide angle in degrees (default 15) or 'continuous' to turn until stopped. Example: '15' or '45' or 'continuous'"
        ),
        Tool(
            name="turn_robot_right",
            func=turn_robot_right,
            description="Turn the robot right. Provide angle in degrees (default 15) or 'continuous' to turn until stopped. Example: '15' or '45' or 'continuous'"
        ),
        Tool(
            name="move_robot_forward",
            func=move_robot_forward,
            description="Move the robot forward. Provide distance in cm (default 10) or 'continuous' to move until stopped. Example: '10' or '50' or 'continuous'"
        ),
        Tool(
            name="move_robot_backward",
            func=move_robot_backward,
            description="Move the robot backward. Provide distance in cm (default 10) or 'continuous' to move until stopped. Example: '10' or '50' or 'continuous'"
        ),
        Tool(
            name="stop_robot_motion",
            func=stop_robot_motion,
            description="Stop all robot motion immediately. Use this to halt continuous movement or turning."
        ),
    ]

