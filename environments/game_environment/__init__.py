"""
Game environment for discrete game interaction.
Provides tools and utilities for controlling and interfacing with a discrete game.
"""

from .game_tools import (
    initialize_game,
    close_game,
    get_game_instance,
    capture_game_image,
    move_game_forward,
    move_game_backward,
    turn_game_clockwise,
    turn_game_counterclockwise,
    get_latest_game_image,
    analyze_current_game_view,
    create_game_tools
)

from .init_game import (
    create_default_game,
    create_game_with_gold,
    create_random_game
)

from .llm_wrapper import inject_image, encode_image_to_base64, encode_numpy_image_to_base64

__all__ = [
    'initialize_game',
    'close_game',
    'get_game_instance',
    'capture_game_image',
    'move_game_forward',
    'move_game_backward',
    'turn_game_clockwise',
    'turn_game_counterclockwise',
    'get_latest_game_image',
    'analyze_current_game_view',
    'create_game_tools',
    'create_default_game',
    'create_game_with_gold',
    'create_random_game',
    'inject_image',
    'encode_image_to_base64',
    'encode_numpy_image_to_base64'
]
