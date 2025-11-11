"""
Sequential thinking tools.
Provides tools for step-by-step reasoning and problem decomposition.
"""

from typing import List

from sequential_thinking_tool import SequentialThinkingTool


# Global thinking tool instance
thinking_tool_instance: SequentialThinkingTool = None


def initialize_thinking_tool():
    """Initialize the global thinking tool instance."""
    global thinking_tool_instance
    if thinking_tool_instance is None:
        thinking_tool_instance = SequentialThinkingTool()
    return thinking_tool_instance


def create_thinking_tools() -> List:
    """
    Create and return thinking tools.
    
    Returns:
        List containing the SequentialThinkingTool instance
    """
    tool = initialize_thinking_tool()
    return [tool]


def get_thinking_history():
    """
    Get the history of sequential thinking steps.
    
    Returns:
        History of all thinking steps executed by the tool
    """
    if thinking_tool_instance is None:
        return []
    return thinking_tool_instance.get_history()

