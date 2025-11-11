"""
Tools package for the Experience Engine agent.
Aggregates all tools from different modules.
"""

from typing import List

from langchain.tools import Tool

from tools.memory_tools import initialize_vector_store_manager, refresh_all_vector_stores
from tools.filesystem_tools import create_filesystem_tools
from tools.semantic_tools import create_semantic_tools
from tools.procedural_tools import create_procedural_tools
from tools.episodic_tools import create_episodic_tools
from tools.thinking_tools import create_thinking_tools


def create_tools() -> List[Tool]:
    """
    Create and return all tools for the agent.
    
    Returns:
        List of all available tools (filesystem + memory + thinking + utility)
    """
    tools = []
    
    # Add filesystem tools
    tools.extend(create_filesystem_tools())
    
    # Add memory query tools
    tools.extend(create_semantic_tools())
    tools.extend(create_procedural_tools())
    tools.extend(create_episodic_tools())
    
    # Add thinking tools
    tools.extend(create_thinking_tools())
    
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
]

