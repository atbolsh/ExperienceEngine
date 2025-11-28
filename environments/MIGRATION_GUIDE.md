# Environment System Migration Guide

## Quick Reference: What Changed

### File Locations

| Component | Before | After |
|-----------|--------|-------|
| Car hardware interface | `utils/car.py` | `environments/car_environment/car.py` |
| Robot control tools | `tools/robot_tools.py` | `environments/car_environment/robot_tools.py` |
| Image injection | `utils/llm_wrapper.py` | `environments/car_environment/llm_wrapper.py` |

### Import Statements

**Before:**
```python
from tools.robot_tools import create_robot_tools, initialize_car, close_car
from utils.llm_wrapper import inject_image
from utils.car import Car
```

**After:**
```python
from environments.car_environment import (
    create_robot_tools, 
    initialize_car, 
    close_car,
    inject_image,
    Car
)
```

## Using the New Environment System

### For End Users

No changes needed! The agent works exactly the same way. The refactoring is entirely internal.

### For Developers

If you have custom scripts that import robot functionality:

1. **Update imports**: Change `tools.robot_tools` → `environments.car_environment`
2. **Update imports**: Change `utils.llm_wrapper` → `environments.car_environment`
3. **Update imports**: Change `utils.car` → `environments.car_environment`

### Examples

#### Capturing Images

```python
# Old way (still works via updated imports)
from tools.robot_tools import capture_robot_image

# New way (cleaner)
from environments.car_environment import capture_robot_image

# Use it
result = capture_robot_image()
```

#### Image Injection

```python
# Old way
from utils.llm_wrapper import inject_image

# New way
from environments.car_environment import inject_image

# Use it
enhanced_input = inject_image("What do you see?")
```

#### Direct Car Control

```python
# Old way
from utils.car import Car

# New way
from environments.car_environment import Car

# Use it
car = Car()
car.start()
car.forward(speed=50)
```

## Switching Environments

To use a different environment in the future:

1. **Edit `select_environment.config`**:
   ```
   ACTIVE_ENVIRONMENT=new_environment
   ```

2. **Update imports in `tools/__init__.py`**:
   ```python
   from environments.new_environment import create_robot_tools, initialize_car, close_car
   ```

3. **Update imports in `main.py`**:
   ```python
   from environments.new_environment import inject_image
   ```

4. **Restart the agent**

## Creating a New Environment

See `environments/ENVIRONMENT_REQUIREMENTS.md` for detailed requirements.

### Quick Start

1. Create directory: `environments/my_environment/`
2. Create required files:
   - `__init__.py` - Package exports
   - `core_module.py` - Core functionality
   - `tools.py` - LangChain tools
   - `llm_wrapper.py` - Image injection
3. Implement required functions:
   - `inject_image(user_input)` - Inject sensory data
   - `capture_[env]_image()` - Capture and view
   - `create_[env]_tools()` - Create tool list
   - `initialize_[env]()` - Setup
   - `close_[env]()` - Cleanup
4. Update `select_environment.config`
5. Update imports in `tools/__init__.py` and `main.py`

## Backward Compatibility

✅ All existing functionality is preserved  
✅ No changes to agent behavior  
✅ No changes to user interface  
✅ All robot tools work the same  
✅ Image injection works the same  

## Troubleshooting

### Import Errors

If you see:
```
ModuleNotFoundError: No module named 'tools.robot_tools'
```

Solution: Update imports to use `environments.car_environment`

### Missing Functions

If you see:
```
ImportError: cannot import name 'function_name' from 'environments.car_environment'
```

Solution: Check `environments/car_environment/__init__.py` to see exported functions

### Environment Not Found

If you see:
```
ModuleNotFoundError: No module named 'environments.car_environment'
```

Solution: Ensure you're running from the project root directory

## Testing Your Changes

```bash
# Test imports
cd /home/atbolsh/Vibe/AIagents/experience-engine
python -c "from environments.car_environment import initialize_car, inject_image; print('Success!')"

# Run the agent
python main.py

# Run tests
python test_image_injection.py
```

## Questions?

- See `environments/ENVIRONMENT_REQUIREMENTS.md` for environment creation details
- See `environments/car_environment/README.md` for car environment specifics
- See `ENVIRONMENT_REFACTORING.md` for refactoring overview

