"""
Game initialization helper.
Call this to initialize the game environment before using game tools.
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Run pygame in headless mode

from game.discreteEngine import discreteGame
from game.levels.skeleton import Settings
from tools.game_tools import initialize_game


def create_default_game():
    """Create a game with default settings."""
    settings = Settings(gameSize=64)
    game = discreteGame(settings=settings, envMode=True)
    initialize_game(game)
    return game


def create_game_with_gold():
    """Create a game with gold pieces and walls."""
    settings = Settings(
        gameSize=64,
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
    initialize_game(game)
    return game


def create_random_game():
    """Create a randomly generated game."""
    game = discreteGame(envMode=True)
    initialize_game(game)
    return game


# Auto-initialize with default game when imported
# This ensures the game is ready to use immediately
print("Initializing game environment...")
create_default_game()
print("Game environment initialized and ready!")

