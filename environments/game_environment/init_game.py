"""
Game initialization helper.
Provides functions to create game instances with specific configurations.
"""

import random
from .discreteEngine import discreteGame


def random_bare_game():
    """Create a game with default settings: 224x224, no internal walls, 1 gold."""
    # Create a temporary game instance to access its helper methods
    temp_game = discreteGame(envMode=True)
    
    # Use random_settings with gameSize=224 as the base
    settings = temp_game.random_settings(gameSize=224, restrict_angles=False)
    
    # Override walls to have no internal walls (only boundary walls)
    settings.walls = temp_game.random_walls(restrict_angles=False, num_extra_walls=0)
    
    # Override gold to have exactly 1 piece
    num_gold = 1
    settings.gold = temp_game.random_gold(settings.walls, max_num_gold=num_gold, 
                                          agent_x=settings.agent_x, agent_y=settings.agent_y)
    
    # Create the game with the modified settings
    game = discreteGame(settings=settings, envMode=True)
    return game


def create_random_two_walls_game():
    """Create a game with 224x224, exactly 2 horizontal/vertical internal walls, 1 gold."""
    # Create a temporary game instance to access its helper methods
    temp_game = discreteGame(envMode=True)
    
    # Use random_settings with gameSize=224 and restrict_angles=True
    settings = temp_game.random_settings(gameSize=224, restrict_angles=True)
    
    # Build walls: boundary walls + exactly 2 internal walls (horizontal/vertical)
    walls = temp_game.random_side_walls()
    walls.append(temp_game.random_wall(restrict_angles=True))
    walls.append(temp_game.random_wall(restrict_angles=True))
    settings.walls = walls
    
    # Override gold to have exactly 1 piece
    num_gold = 1
    settings.gold = temp_game.random_gold(settings.walls, max_num_gold=num_gold, 
                                          agent_x=settings.agent_x, agent_y=settings.agent_y)
    
    # Create the game with the modified settings
    game = discreteGame(settings=settings, envMode=True)
    return game


# Default game creation function - change this to switch between different game modes
default_game = create_random_two_walls_game


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

