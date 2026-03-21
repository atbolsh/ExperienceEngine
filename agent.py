"""
Agent creation and configuration.
Handles the creation of the AgentExecutor with tools and memory.
"""

import os
from pathlib import Path
from typing import Optional

# LangChain >=1.0 (e.g. 1.2.x): agents moved to langchain-classic; prompts often live in langchain-core.
try:
    from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
except ImportError:  # pragma: no cover - depends on installed langchain major version
    from langchain_classic.agents import (
        AgentExecutor,
        create_react_agent,
        create_structured_chat_agent,
    )

try:
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:  # pragma: no cover
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from active_environment import get_active_environment_name
from tools import create_tools, initialize_vector_store_manager, initialize_env
from tools.memory_schema import list_memory_folders
from llm_utils import get_local_llm

STRUCTURED_CHAT_SUFFIX_PATH = Path(__file__).resolve().parent / "prompts" / "structured_chat_system_suffix.txt"
STRUCTURED_CHAT_HUB_ID = "hwchase17/structured-chat-agent"
REACT_AGENT_SUFFIX_PATH = Path(__file__).resolve().parent / "prompts" / "react_agent_suffix.txt"


def load_react_agent_suffix() -> str:
    """ReAct tool-calling instructions ({tools}, {tool_names}) for small local LLMs."""
    try:
        text = REACT_AGENT_SUFFIX_PATH.read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError as e:
        print(f"[Agent] Missing {REACT_AGENT_SUFFIX_PATH.name} ({e}); using minimal ReAct suffix.")

    return (
        "Tools:\n{tools}\n\nTool names: {tool_names}\n\n"
        "Use: Thought: ... then Action: <name> then Action Input: <one line, JSON if needed> "
        "or Thought: ... then Final Answer: <reply>.\n"
        "Do not write Observation yourself.\n"
    )


def _structured_chat_suffix_from_hub_pull(pulled) -> str:
    """Reduce hub.pull(...) to a single system string with {tools} / {tool_names} if present."""
    from langchain_core.prompts import ChatPromptTemplate

    if isinstance(pulled, ChatPromptTemplate):
        chunks: list[str] = []
        for msg in pulled.messages:
            prompt = getattr(msg, "prompt", None)
            tmpl = getattr(prompt, "template", None) if prompt is not None else None
            if isinstance(tmpl, str) and tmpl.strip():
                chunks.append(tmpl.strip())
        if chunks:
            return "\n\n".join(chunks)

    tmpl = getattr(pulled, "template", None)
    if isinstance(tmpl, str) and tmpl.strip():
        return tmpl.strip()

    raise TypeError(f"Unsupported hub artifact type: {type(pulled)!r}")


def load_structured_chat_system_suffix() -> str:
    """
    System suffix for structured chat (JSON tool protocol and {tools} / {tool_names}).

    Primary: ``prompts/structured_chat_system_suffix.txt`` (canonical copy in repo).
    Fallback: ``langchain_classic.hub.pull("hwchase17/structured-chat-agent")``.
    """
    try:
        text = STRUCTURED_CHAT_SUFFIX_PATH.read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError as e:
        print(f"[Agent] Missing {STRUCTURED_CHAT_SUFFIX_PATH.name} ({e}); trying langchain_classic.hub...")

    try:
        from langchain_classic import hub

        pulled = hub.pull(STRUCTURED_CHAT_HUB_ID)
        suffix = _structured_chat_suffix_from_hub_pull(pulled)
        if suffix.strip():
            print(f"[Agent] Loaded structured-chat suffix from Hub fallback ({STRUCTURED_CHAT_HUB_ID}).")
            return suffix
    except Exception as e:
        print(f"[Agent] langchain_classic.hub fallback failed: {e}")

    raise RuntimeError(
        f"Could not load structured chat system suffix. Restore prompts/{STRUCTURED_CHAT_SUFFIX_PATH.name} "
        f"or install langchain-classic and allow Hub access for {STRUCTURED_CHAT_HUB_ID!r}."
    )


def load_environment_blurb() -> str:
    """
    Load the environment-specific blurb (same resolution as tools / active_environment).
    """
    env_name = get_active_environment_name()

    if env_name == "car_environment":
        blurb_file = "car_blurb.txt"
    elif env_name == "game_environment":
        blurb_file = "game_blurb.txt"
    else:
        print(f"Unknown environment '{env_name}', defaulting to game_blurb.txt")
        blurb_file = "game_blurb.txt"
    
    # Load the blurb file
    blurb_path = os.path.join(os.path.dirname(__file__), 'prompts', blurb_file)
    try:
        with open(blurb_path, 'r') as f:
            blurb_text = f.read()
        print(f"[Agent] Loaded environment blurb: {blurb_file}")
        return blurb_text
    except FileNotFoundError:
        print(f"Warning: Blurb file not found: {blurb_path}")
        return "Environment interface information not available."


def load_context_prompt() -> str:
    """
    Load the context_prompt.md file containing learned hints.
    
    Returns:
        The context prompt text or empty string if not found
    """
    context_path = Path(__file__).parent / "prompts" / "context_prompt.md"
    try:
        if context_path.exists():
            with open(context_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"[Agent] Loaded context_prompt.md")
            return content
        else:
            print("[Agent] No context_prompt.md found (will be created when agent edits it)")
            return ""
    except Exception as e:
        print(f"Warning: Error loading context_prompt.md: {e}")
        return ""


def load_most_recent_episodic_memory() -> str:
    """
    Load the most recent episodic memory to provide continuity.
    
    Returns:
        The most recent episodic memory content or empty string if none found
    """
    try:
        episodic_path = Path(__file__).parent / "episodic"
        if not episodic_path.exists():
            print("[Agent] No episodic memory folder found")
            return ""
        
        # Get all memory folders
        memories = list_memory_folders(episodic_path)
        
        if not memories:
            print("[Agent] No episodic memories found")
            return ""
        
        # Sort by modification time (most recent first)
        # Use key parameter to avoid comparing MemoryFolder objects if timestamps are equal
        most_recent = max(memories, key=lambda m: m.folder_path.stat().st_mtime)
        
        print(f"[Agent] Loaded most recent episodic memory: {most_recent.folder_name}")
        
        # Format the memory
        result = f"=== MOST RECENT EPISODIC MEMORY ===\n"
        result += f"Memory: {most_recent.folder_name}\n\n"
        result += most_recent.text_content
        
        if most_recent.image_files:
            result += f"\n\n(This memory contains {len(most_recent.image_files)} images: {', '.join(most_recent.image_names)})"
        
        result += "\n=== END OF RECENT MEMORY ===\n"
        
        return result
        
    except Exception as e:
        print(f"Warning: Error loading most recent episodic memory: {e}")
        return ""


def create_agent():
    """Create and configure the agent with tools and memory."""
    
    # Initialize vector store manager
    print("Initializing vector stores...")
    initialize_vector_store_manager()
    
    # Initialize environment (car, game, etc.)
    initialize_env()
    
    # Create the LLM using local Qwen3 0.6B model
    # Image injection still happens in main.py at input level
    # Vision-based tool calls use local Qwen2-VL via llm_utils.get_vision_llm()
    llm = get_local_llm()
    
    # Create tools
    tools = create_tools()
    
    # Load the prompt from the external file
    prompt_file_path = os.path.join(os.path.dirname(__file__), "prompts", "global_prompt.txt")
    with open(prompt_file_path, 'r') as f:
        prompt_text = f.read()
    
    # Load environment-specific blurb
    environment_blurb = load_environment_blurb()
    
    # Load context prompt (learned hints)
    context_prompt = load_context_prompt()
    
    # Load most recent episodic memory for continuity
    recent_memory = load_most_recent_episodic_memory()
    
    # Build the continuity section
    continuity_section = ""
    
    if context_prompt:
        continuity_section += "\n\n**LEARNED CONTEXT (from context_prompt.md):**\n"
        continuity_section += context_prompt
    
    if recent_memory:
        continuity_section += "\n\n**SESSION CONTINUITY:**\n"
        continuity_section += "To help you understand what happened in the previous session, here is the most recent episodic memory:\n\n"
        continuity_section += recent_memory
    
    # Format the prompt with cwd, environment_blurb, and continuity
    formatted_prompt = prompt_text.format(
        cwd=os.getcwd(),
        environment_blurb=environment_blurb
    )
    
    # Append continuity section to the end of the prompt
    if continuity_section:
        formatted_prompt += continuity_section
    
    # Qwen3-0.6B rarely emits valid structured-chat JSON; ReAct (Thought/Action/Action Input) parses more reliably.
    # Opt into JSON structured chat with USE_STRUCTURED_CHAT_AGENT=1 (stronger models only).
    use_structured = os.environ.get("USE_STRUCTURED_CHAT_AGENT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if use_structured:
        print("[Agent] Structured-chat (JSON) agent — USE_STRUCTURED_CHAT_AGENT is set.")
        tool_suffix = load_structured_chat_system_suffix()
        human_tail = "{input}\n\n{agent_scratchpad}"
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", formatted_prompt + "\n\n" + tool_suffix),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", human_tail),
            ]
        )
        agent = create_structured_chat_agent(llm, tools, prompt)
    else:
        print("[Agent] ReAct agent (default for local Qwen3-0.6B). Set USE_STRUCTURED_CHAT_AGENT=1 for JSON tools.")
        tool_suffix = load_react_agent_suffix()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", formatted_prompt + "\n\n" + tool_suffix),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "Question: {input}\nThought:{agent_scratchpad}"),
            ]
        )
        # Some chat pipelines ignore or mishandle stop sequences; REACT_AGENT_NO_STOP=1 disables \nObservation stop.
        no_stop = os.environ.get("REACT_AGENT_NO_STOP", "").strip().lower() in ("1", "true", "yes")
        agent = create_react_agent(llm, tools, prompt, stop_sequence=not no_stop)

    parse_hint = (
        "Format error. Use: Thought: ... then Action: <exact tool name> then Action Input: <one line> "
        "OR Thought: ... then Final Answer: <your reply>. Do not write Observation."
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=parse_hint,
        max_iterations=45,
    )
    
    return agent_executor


def create_conversational_agent():
    """Create an agent with conversation memory."""
    
    # Create the base agent
    agent_executor = create_agent()
    
    # Create message history
    message_history = ChatMessageHistory()
    
    # Wrap the agent with message history
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return agent_with_chat_history, message_history
