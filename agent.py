"""
Agent creation and configuration.
Handles the creation of the AgentExecutor with tools and memory.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent

try:
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from active_environment import get_active_environment_name
from qwen3_react_parser import Qwen3ReActOutputParser
from tools import create_tools, initialize_vector_store_manager, initialize_env
from tools.memory_schema import list_memory_folders
from llm_utils import get_local_llm

REACT_SUFFIX_PATH = Path(__file__).resolve().parent / "prompts" / "react_agent_suffix.txt"


def _load_react_suffix() -> str:
    try:
        text = REACT_SUFFIX_PATH.read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    return (
        "Tools:\n{tools}\n\nTool names: {tool_names}\n\n"
        "Thought: ... then Action: <name> then Action Input: <input>\n"
        "or Thought: ... then Final Answer: <reply>\n"
    )


def load_environment_blurb() -> str:
    env_name = get_active_environment_name()
    name_to_file = {
        "car_environment": "car_blurb.txt",
        "game_environment": "game_blurb.txt",
    }
    blurb_file = name_to_file.get(env_name, "game_blurb.txt")
    blurb_path = Path(__file__).resolve().parent / "prompts" / blurb_file
    try:
        text = blurb_path.read_text(encoding="utf-8")
        print(f"[Agent] Loaded environment blurb: {blurb_file}")
        return text
    except FileNotFoundError:
        print(f"Warning: Blurb file not found: {blurb_path}")
        return ""


def _load_optional_text(path: Path, label: str) -> str:
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                print(f"[Agent] Loaded {label}")
                return text
    except Exception as e:
        print(f"Warning loading {label}: {e}")
    return ""


def load_context_prompt() -> str:
    return _load_optional_text(
        Path(__file__).parent / "prompts" / "context_prompt.md",
        "context_prompt.md",
    )


def load_most_recent_episodic_memory() -> str:
    try:
        ep = Path(__file__).parent / "episodic"
        if not ep.exists():
            return ""
        memories = list_memory_folders(ep)
        if not memories:
            return ""
        most_recent = max(memories, key=lambda m: m.folder_path.stat().st_mtime)
        print(f"[Agent] Loaded episodic memory: {most_recent.folder_name}")
        result = f"=== RECENT EPISODIC MEMORY ===\n{most_recent.folder_name}\n\n"
        result += most_recent.text_content
        if most_recent.image_files:
            result += f"\n({len(most_recent.image_files)} images: {', '.join(most_recent.image_names)})"
        result += "\n=== END ===\n"
        return result
    except Exception as e:
        print(f"Warning: episodic memory: {e}")
        return ""


def create_agent():
    """Create and configure the agent with tools and memory."""
    print("Initializing vector stores...")
    initialize_vector_store_manager()
    initialize_env()

    llm = get_local_llm()
    tools = create_tools()

    # ── system prompt ────────────────────────────────────────────────
    prompt_path = Path(__file__).resolve().parent / "prompts" / "global_prompt.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")

    environment_blurb = load_environment_blurb()
    context_prompt = load_context_prompt()
    recent_memory = load_most_recent_episodic_memory()

    formatted = prompt_text.format(
        cwd=os.getcwd(),
        environment_blurb=environment_blurb,
    )

    extras: list[str] = []
    if context_prompt:
        extras.append(f"Learned context:\n{context_prompt}")
    if recent_memory:
        extras.append(f"Previous session:\n{recent_memory}")
    if extras:
        formatted += "\n\n" + "\n\n".join(extras)

    tool_suffix = _load_react_suffix()
    system_msg = formatted + "\n\n" + tool_suffix

    # ── prompt template ──────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}\nThought:{agent_scratchpad}"),
        ]
    )

    # ── agent ────────────────────────────────────────────────────────
    parser = Qwen3ReActOutputParser()
    agent = create_react_agent(llm, tools, prompt, output_parser=parser)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=(
            "Format error. Reply with EXACTLY:\n"
            "Thought: <reasoning>\n"
            "Action: <tool_name>\n"
            "Action Input: <input>\n\n"
            "or:\nThought: <reasoning>\nFinal Answer: <reply>"
        ),
        max_iterations=30,
    )
    return agent_executor


def create_conversational_agent():
    """Create an agent with conversation memory."""
    agent_executor = create_agent()
    message_history = ChatMessageHistory()
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history, message_history
