"""
Tools package for the Experience Engine agent.
Aggregates all tools from different modules and loads environment dynamically.
"""

import sys
import os
from typing import List, Tuple, Any

from langchain.tools import Tool

from tools.memory_tools import initialize_vector_store_manager, refresh_all_vector_stores, clear_working_memory_function
from tools.filesystem_tools import create_filesystem_tools
from tools.semantic_tools import create_semantic_tools
from tools.procedural_tools import create_procedural_tools
from tools.episodic_tools import create_episodic_tools
from tools.working_tools import create_working_tools
from tools.thinking_tools import create_thinking_tools
from tools.scripting_tools import create_scripting_tools
from tools.image_tools import create_image_tools
from tools.memory_writing_tools import create_memory_writing_tools
from tools.session_tools import create_session_tools
from tools.tool_writing_tools import create_tool_writing_tools
from tools.image_processing_tools import create_image_processing_tools


def load_environment_config() -> str:
    """
    Load the active environment from select_environment.config.
    
    Returns:
        The name of the active environment (e.g., 'car_environment', 'game_environment')
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'select_environment.config')
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Defaulting to car_environment.")
        return 'car_environment'
    
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('ACTIVE_ENVIRONMENT='):
                    env_name = line.split('=', 1)[1].strip()
                    print(f"[Environment Loader] Selected environment: {env_name}")
                    return env_name
        
        print("Warning: ACTIVE_ENVIRONMENT not found in config. Defaulting to car_environment.")
        return 'car_environment'
    except Exception as e:
        print(f"Error reading config file: {e}. Defaulting to car_environment.")
        return 'car_environment'


def load_environment() -> Tuple[Any, Any, Any]:
    """
    Dynamically load the active environment based on select_environment.config.
    
    Returns:
        Tuple of (create_tools_func, initialize_func, close_func)
    """
    env_name = load_environment_config()
    
    try:
        if env_name == 'car_environment':
            from environments.car_environment import create_robot_tools, initialize_car, close_car
            return create_robot_tools, initialize_car, close_car
        elif env_name == 'game_environment':
            from environments.game_environment import create_game_tools, initialize_game, close_game
            return create_game_tools, initialize_game, close_game
        else:
            print(f"Warning: Unknown environment '{env_name}'. Defaulting to car_environment.")
            from environments.car_environment import create_robot_tools, initialize_car, close_car
            return create_robot_tools, initialize_car, close_car
    except ImportError as e:
        print(f"Error importing environment '{env_name}': {e}")
        print("Falling back to car_environment.")
        from environments.car_environment import create_robot_tools, initialize_car, close_car
        return create_robot_tools, initialize_car, close_car


# Load the active environment
create_env_tools, initialize_env, close_env = load_environment()


def create_tools() -> List[Tool]:
    """
    Create and return all tools for the agent.
    
    Returns:
        List of all available tools (filesystem + memory + scripting + thinking + utility + environment)
    """
    tools = []
    
    # Add filesystem tools
    tools.extend(create_filesystem_tools())
    
    # Add memory query tools
    tools.extend(create_semantic_tools())
    tools.extend(create_procedural_tools())
    tools.extend(create_episodic_tools())
    tools.extend(create_working_tools())
    
    # Add memory writing tools
    tools.extend(create_memory_writing_tools())
    
    # Add scripting tools
    tools.extend(create_scripting_tools())
    
    # Add image tools
    tools.extend(create_image_tools())
    
    # Add thinking tools
    tools.extend(create_thinking_tools())
    
    # Add tool writing capability
    tools.extend(create_tool_writing_tools())
    
    # Add session control tools
    tools.extend(create_session_tools())
    
    # === CUSTOM TOOLS SECTION (auto-generated tools will be added below) ===
    # Add custom tool: draw_line_to_closest_dot
    tools.extend(create_image_processing_tools())
    
    # Add environment-specific tools (loaded dynamically)
    tools.extend(create_env_tools())
    
    # Add utility tool for refreshing vector stores
    tools.append(
        Tool(
            name="refresh_vector_stores",
            func=refresh_all_vector_stores,
            description="Refresh all vector stores to pick up new or modified documents. Use after adding or modifying files in semantic, procedural, or episodic directories. No input required.",
        )
    )
    
    return tools


# Export commonly used functions
__all__ = [
    'create_tools',
    'initialize_vector_store_manager',
    'clear_working_memory_function',
    'initialize_env',
    'close_env',
]
