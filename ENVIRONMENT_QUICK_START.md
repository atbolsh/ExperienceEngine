# Quick Start: Switching Environments

## Current Setup

The Experience Engine now supports **two environments**:

1. **car_environment** - Physical robot car with camera
2. **game_environment** - Discrete 2D game simulation

## How to Switch

### Step 1: Edit Configuration File

Open `select_environment.config` and change the `ACTIVE_ENVIRONMENT` line:

**For Robot Car:**
```
ACTIVE_ENVIRONMENT=car_environment
```

**For Game:**
```
ACTIVE_ENVIRONMENT=game_environment
```

### Step 2: Start the Agent

```bash
python main.py
```

That's it! The system automatically:
- Loads the correct environment
- Initializes the appropriate tools
- Injects the right type of images (camera or game screenshots)

## Quick Test

### Test Car Environment

1. Set `ACTIVE_ENVIRONMENT=car_environment`
2. Run `python main.py`
3. Try these commands:
   ```
   You: capture_robot_image
   You: What do you see?
   You: move forward
   You: turn left
   ```

### Test Game Environment

1. Set `ACTIVE_ENVIRONMENT=game_environment`
2. Run `python main.py`
3. Try these commands:
   ```
   You: capture_game_image
   You: What do you see in the game?
   You: move forward
   You: turn clockwise
   ```

## Tool Comparison

| Car Environment | Game Environment |
|----------------|------------------|
| `capture_robot_image` | `capture_game_image` |
| `analyze_current_view` | `analyze_current_game_view` |
| `move_robot_forward` | `move_game_forward` |
| `move_robot_backward` | `move_game_backward` |
| `turn_robot_left` | `turn_game_counterclockwise` |
| `turn_robot_right` | `turn_game_clockwise` |
| `stop_robot_motion` | (not needed - game uses discrete steps) |

## Troubleshooting

### "ModuleNotFoundError: No module named 'environments.game_environment'"

Solution: Make sure the game files are in the correct location:
```
environments/game_environment/
├── discreteEngine.py
├── game_tools.py
├── llm_wrapper.py
└── ...
```

### "Robot car not initialized" when using game environment

Solution: Check `select_environment.config` - make sure it says `game_environment` not `car_environment`

### Images not showing up

Solution: Both environments automatically capture and inject images. Use the capture tools to ensure images are being saved to the `working/` directory.

## Using init_game.py

For custom game initialization:

```python
# Option 1: Use init_game.py helper
from init_game import create_game_with_gold, create_random_game

game = create_game_with_gold()  # Pre-configured game
# or
game = create_random_game()  # Random level

# Option 2: Direct initialization
from environments.game_environment.discreteEngine import discreteGame
from environments.game_environment.levels.skeleton import Settings
from environments.game_environment import initialize_game

settings = Settings(gameSize=64, ...)
game = discreteGame(settings=settings, envMode=True)
initialize_game(game)
```

## File Locations

- Configuration: `select_environment.config`
- Car environment: `environments/car_environment/`
- Game environment: `environments/game_environment/`
- Game initialization helper: `init_game.py`
- Working images: `working/`

## Documentation

- Full environment requirements: `environments/ENVIRONMENT_REQUIREMENTS.md`
- Car environment details: `environments/car_environment/README.md`
- Game environment details: `environments/game_environment/README.md`
- Integration summary: `GAME_ENVIRONMENT_INTEGRATION.md`

