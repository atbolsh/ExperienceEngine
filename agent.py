"""
Agent creation and configuration.
Handles the creation of the AgentExecutor with tools and memory.
"""

import os
from pathlib import Path
from typing import Optional

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub

from tools import create_tools, initialize_vector_store_manager, initialize_env
from tools.memory_schema import list_memory_folders
from llm_utils import get_local_llm


def load_environment_blurb() -> str:
    """
    Load the environment-specific blurb based on select_environment.config.
    
    Returns:
        The environment blurb text
    """
    config_path = os.path.join(os.path.dirname(__file__), 'select_environment.config')
    env_name = 'car_environment'  # Default
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('ACTIVE_ENVIRONMENT='):
                        env_name = line.split('=', 1)[1].strip()
                        break
        except Exception as e:
            print(f"Error reading config: {e}")
    
    # Determine which blurb file to load
    if env_name == 'car_environment':
        blurb_file = 'car_blurb.txt'
    elif env_name == 'game_environment':
        blurb_file = 'game_blurb.txt'
    else:
        print(f"Unknown environment '{env_name}', defaulting to car_blurb.txt")
        blurb_file = 'car_blurb.txt'
    
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
    # Vision-based tool calls will use GPT-5 separately
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
    
    # For local models like Qwen3 0.6B, we use a structured chat agent
    # which works better with smaller models that don't have native tool calling
    # Pull the structured chat prompt from LangChain hub and customize it
    base_prompt = hub.pull("hwchase17/structured-chat-agent")
    
    # Inject our system prompt into the structured chat template
    structured_system_prefix = formatted_prompt + """

You have access to the following tools. Use them wisely to accomplish tasks.

IMPORTANT: When you want to use a tool, respond with a JSON blob with "action" and "action_input" keys.
The "action" value must be the exact tool name, and "action_input" must be a dict with the tool's parameters.

Example tool use:
```json
{{
  "action": "list_directory",
  "action_input": {{"directory_path": "."}}
}}
```

When you have gathered enough information or completed the task, respond with:
```json
{{
  "action": "Final Answer",
  "action_input": "Your final response here"
}}
```

Available tools:
{tools}

Tool names: {tool_names}
"""
    
    # Create the prompt template with our custom system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", structured_system_prefix),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}\n\n{agent_scratchpad}"),
    ])
    
    # Create the structured chat agent (works better with local models)
    agent = create_structured_chat_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
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
