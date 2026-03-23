"""
Agent dispatcher.

Selects the agent backend based on the AGENT_TYPE environment variable:

    AGENT_TYPE=nous      (default)  Qwen3 native <tool_call> protocol
    AGENT_TYPE=langchain            LangChain ReAct agent (AgentExecutor)

Both backends expose the same public API:

    agent, history = create_conversational_agent()
    result = agent.invoke({"input": "..."}, config={"configurable": {"session_id": "s1"}})
    print(result["output"])
    history.clear()
"""

import os

# Re-export shared helpers so existing imports like
# ``from agent import load_context_prompt`` keep working.
from nous_agent import (              # noqa: F401  – re-exports
    _build_system_prompt as _build_system_prompt,
    _load_optional_text as _load_optional_text,
)

def load_context_prompt():
    from nous_agent import _load_optional_text
    from pathlib import Path
    return _load_optional_text(
        Path(__file__).parent / "prompts" / "context_prompt.md",
        "context_prompt.md",
    )

def load_most_recent_episodic_memory():
    from nous_agent import _load_recent_episodic
    return _load_recent_episodic()


def _agent_type() -> str:
    return os.environ.get("AGENT_TYPE", "nous").strip().lower()


def create_conversational_agent():
    kind = _agent_type()
    if kind == "langchain":
        print(f"[Agent] Using LangChain ReAct backend (AGENT_TYPE={kind})")
        from langchain_agent import create_conversational_agent as _create
    else:
        print(f"[Agent] Using Nous/Qwen3 native backend (AGENT_TYPE={kind})")
        from nous_agent import create_conversational_agent as _create
    return _create()
