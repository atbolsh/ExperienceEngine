"""
Game initialization helper.
Provides functions to create game instances with specific configurations.
"""

import random
from .discreteEngine import discreteGame


def random_bare_game():
    """Create a game with default settings: 224x224, no internal walls, 1 gold."""
    from .levels.skeleton import Settings

    helper = discreteGame(envMode=True)
    walls = helper.random_side_walls()
    agent_x, agent_y = helper.random_valid_coords(walls, helper.typical_agent_r)
    gold = helper.random_gold(walls, max_num_gold=1, agent_x=agent_x, agent_y=agent_y)

    import random, math
    settings = Settings(
        gameSize=224,
        agent_r=helper.typical_agent_r,
        gold_r=helper.typical_gold_r,
        walls=walls,
        gold=gold,
        agent_x=agent_x,
        agent_y=agent_y,
        direction=random.uniform(0, 2 * math.pi),
    )
    return discreteGame(settings=settings, envMode=True)


def create_random_two_walls_game():
    """Create a game with 224x224, exactly 2 horizontal/vertical internal walls, 1 gold."""
    from .levels.skeleton import Settings

    helper = discreteGame(envMode=True)
    walls = helper.random_side_walls()
    walls.append(helper.random_wall(restrict_angles=True))
    walls.append(helper.random_wall(restrict_angles=True))

    agent_x, agent_y = helper.random_valid_coords(walls, helper.typical_agent_r)
    gold = helper.random_gold(walls, max_num_gold=1, agent_x=agent_x, agent_y=agent_y)

    import random, math
    settings = Settings(
        gameSize=224,
        agent_r=helper.typical_agent_r,
        gold_r=helper.typical_gold_r,
        walls=walls,
        gold=gold,
        agent_x=agent_x,
        agent_y=agent_y,
        direction=random.uniform(0, 2 * math.pi),
    )
    return discreteGame(settings=settings, envMode=True)


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

