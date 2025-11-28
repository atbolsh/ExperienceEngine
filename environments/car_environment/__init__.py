"""
Car environment for robot car interaction.
Provides tools and utilities for controlling and interfacing with a robot car.
"""

from .robot_tools import (
    initialize_car,
    close_car,
    get_car_instance,
    capture_robot_image,
    turn_robot_left,
    turn_robot_right,
    move_robot_forward,
    move_robot_backward,
    stop_robot_motion,
    get_latest_robot_image,
    analyze_current_view,
    create_robot_tools
)

from .llm_wrapper import inject_image, encode_image_to_base64, encode_numpy_image_to_base64

from .car import Car

__all__ = [
    'Car',
    'initialize_car',
    'close_car',
    'get_car_instance',
    'capture_robot_image',
    'turn_robot_left',
    'turn_robot_right',
    'move_robot_forward',
    'move_robot_backward',
    'stop_robot_motion',
    'get_latest_robot_image',
    'analyze_current_view',
    'create_robot_tools',
    'inject_image',
    'encode_image_to_base64',
    'encode_numpy_image_to_base64'
]

