"""
Session control tools for the agent.
Allows the agent to control session lifecycle (end, restart).
"""

from typing import List
from langchain_compat import Tool


# Global flag to signal session control
_session_control_signal = {"action": None}


def get_session_control_signal():
    """Get the current session control signal."""
    return _session_control_signal


def reset_session_control_signal():
    """Reset the session control signal."""
    _session_control_signal["action"] = None


def end_conversation_tool(input_str: str = "") -> str:
    """
    End the current conversation session.
    
    Args:
        input_str: Optional reason for ending the session
    
    Returns:
        Confirmation message
    """
    _session_control_signal["action"] = "end"
    reason = input_str.strip() if input_str.strip() else "Agent decided to end the session"
    return f"Session will end. Reason: {reason}"


def restart_conversation_tool(input_str: str = "") -> str:
    """
    Restart the conversation session (end and immediately start fresh).
    
    Args:
        input_str: Optional reason for restarting
    
    Returns:
        Confirmation message
    """
    _session_control_signal["action"] = "restart"
    reason = input_str.strip() if input_str.strip() else "Agent decided to restart the session"
    return f"Session will restart. Reason: {reason}"


def create_session_tools() -> List[Tool]:
    """
    Create session control tools.
    
    Returns:
        List of session control tools
    """
    return [
        Tool(
            name="end_conversation",
            func=end_conversation_tool,
            description=(
                "End the current conversation session. Use this when you believe the session goals have been "
                "completed, or when it's appropriate to close. Provide an optional reason for ending. "
                "Before using this tool, make sure to record any important memories from the session."
            )
        ),
        Tool(
            name="restart_conversation",
            func=restart_conversation_tool,
            description=(
                "Restart the conversation session (clear history and start fresh). Use this when a fresh start "
                "would be beneficial, such as after completing a major task or when the conversation context "
                "needs to be reset. Provide an optional reason for restarting. "
                "Before using this tool, make sure to record any important memories from the session."
            )
        ),
    ]

