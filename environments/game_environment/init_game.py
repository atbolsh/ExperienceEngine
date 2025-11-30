"""
Game initialization helper.
Provides functions to create game instances with specific configurations.
"""

import random
from .discreteEngine import discreteGame


def create_default_game():
    """Create a game with the tool_use_advanced_2_5 level."""
    # Load the specific level from skeleton.py
    from .levels.skeleton import tool_use_advanced_2_5
    
    # Create the game with the predefined level settings
    game = discreteGame(settings=tool_use_advanced_2_5, envMode=True)
    return game


def create_game_with_gold():
    """Create a game with gold pieces and walls."""
    from .levels.skeleton import Settings
    settings = Settings(
        gameSize=224,
        agent_x=0.5,
        agent_y=0.5,
        agent_r=0.05,
        gold_r=0.015,
        gold=[[0.3, 0.3], [0.7, 0.7]],
        walls=[
            [0, 0, 0.0625, 1.0, 0],  # Left wall
            [0, 0, 1.0, 0.0625, 0],  # Top wall
            [0, 0.9375, 1.0, 0.0625, 0],  # Bottom wall
            [0.9375, 0, 0.0625, 1.0, 0],  # Right wall
        ]
    )
    game = discreteGame(settings=settings, envMode=True)
    return game


def create_random_game():
    """Create a randomly generated game with 224x224 size."""
    # Create a temporary game instance to access random_settings
    temp_game = discreteGame(envMode=True)
    settings = temp_game.random_settings(gameSize=224, restrict_angles=False)
    game = discreteGame(settings=settings, envMode=True)
    return game

