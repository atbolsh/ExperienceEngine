# Dynamic Environment Prompts Update

## Overview

Updated the prompt system to dynamically load environment-specific information, so the agent understands which environment it's in and what tools are available.

## Changes Made

### 1. Updated `prompts/global_prompt.txt`

**Before:**
- Had hardcoded section describing robot car capabilities
- Agent wouldn't know about game environment if loaded

**After:**
- Generic section: "Environment Interface: You are embodied in an environment..."
- Placeholder: `{environment_blurb}` that gets replaced with environment-specific info
- Works for any environment

### 2. Created `prompts/car_blurb.txt`

Environment-specific information for the robot car:
- Camera system and capture tools
- Movement controls (forward, backward, left, right, stop)
- Continuous vs default modes
- Image injection behavior
- Physical robot context

### 3. Created `prompts/game_blurb.txt`

Environment-specific information for the game:
- Game state capture tools
- Movement controls (forward, backward, turn clockwise/counterclockwise)
- **Game elements description**:
  - Green agent with red eye indicator
  - Gold pieces (objective to collect)
  - Black walls (obstacles to navigate)
  - Reward system
- Top-down 64x64 pixel view
- Image injection behavior

### 4. Updated `agent.py`

Added `load_environment_blurb()` function:
- Reads `select_environment.config`
- Determines which environment is active
- Loads corresponding blurb file (`car_blurb.txt` or `game_blurb.txt`)
- Falls back to car if unknown environment

Modified `create_agent()`:
- Calls `load_environment_blurb()`
- Formats prompt with both `{cwd}` and `{environment_blurb}`
- Agent now receives environment-specific context

## How It Works

### Loading Flow

```
1. agent.py starts creating agent
2. load_environment_blurb() is called
3. Reads select_environment.config
4. Determines active environment (car or game)
5. Loads prompts/car_blurb.txt or prompts/game_blurb.txt
6. Inserts blurb into global_prompt.txt at {environment_blurb}
7. Agent receives complete prompt with environment context
```

### Example Prompt for Car Environment

```
2. **Environment Interface**: You are embodied in an environment that you can interact with through
   specialized tools. The environment provides visual feedback through images that are automatically
   attached to your queries.
   
   **Robot Car Interface**: You control a physical robot car with:
   - Camera System: capture_robot_image captures the current view...
   [etc]
```

### Example Prompt for Game Environment

```
2. **Environment Interface**: You are embodied in an environment that you can interact with through
   specialized tools. The environment provides visual feedback through images that are automatically
   attached to your queries.
   
   **Game Environment Interface**: You control an agent in a discrete 2D game world with:
   - Camera System: capture_game_image captures the current state...
   **Game Elements**:
   - Your Agent: A green circle with a red "eye" showing which direction you're facing
   - Gold: Yellow/gold colored circles - your objective is to collect these...
   [etc]
```

## Benefits

1. **Agent Understanding**: Agent now knows:
   - Which environment it's in
   - What the visual elements mean (green agent, red eye, gold, walls)
   - What tools are available
   - What its objective is (in game: collect gold)

2. **No Confusion**: Agent won't try to use car tools in game environment or vice versa

3. **Extensible**: Easy to add new environments - just create `new_env_blurb.txt`

4. **Automatic**: No manual configuration needed - follows `select_environment.config`

## Files Modified

1. `prompts/global_prompt.txt` - Generic environment section with placeholder
2. `prompts/car_blurb.txt` - **NEW** - Car-specific context
3. `prompts/game_blurb.txt` - **NEW** - Game-specific context  
4. `agent.py` - Dynamic blurb loading

## Testing

### Test Car Environment Prompt
1. Set `ACTIVE_ENVIRONMENT=car_environment`
2. Run `python main.py`
3. Agent should understand it controls a physical robot car

### Test Game Environment Prompt
1. Set `ACTIVE_ENVIRONMENT=game_environment`
2. Run `python main.py`
3. Agent should understand:
   - It's a green circle with red eye
   - Objective is to collect gold
   - Black walls are obstacles
   - Movement is discrete (not continuous)

## Verification

Check the agent initialization output:
```
Initializing vector stores...
[Environment Loader] Selected environment: game_environment
Initializing game environment...
Game environment initialized successfully.
[Agent] Loaded environment blurb: game_blurb.txt
```

The agent now has the correct context for whichever environment is active!

---

**Update Complete**: The agent will no longer be confused about which environment it's in. It receives environment-specific instructions dynamically based on the configuration.

