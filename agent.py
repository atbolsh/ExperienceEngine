"""
Agent creation and configuration.
Handles the creation of the AgentExecutor with tools and memory.
"""

import os

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from tools import create_tools, initialize_vector_store_manager, initialize_env


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


def create_agent():
    """Create and configure the agent with tools and memory."""
    
    # Initialize vector store manager
    print("Initializing vector stores...")
    initialize_vector_store_manager()
    
    # Initialize environment (car, game, etc.)
    initialize_env()
    
    # Create the LLM (image injection happens in main.py at input level)
    llm = ChatOpenAI(
        model="gpt-5",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create tools
    tools = create_tools()
    
    # Load the prompt from the external file
    prompt_file_path = os.path.join(os.path.dirname(__file__), "prompts", "global_prompt.txt")
    with open(prompt_file_path, 'r') as f:
        prompt_text = f.read()
    
    # Load environment-specific blurb
    environment_blurb = load_environment_blurb()
    
    # Format the prompt with both cwd and environment_blurb
    formatted_prompt = prompt_text.format(
        cwd=os.getcwd(),
        environment_blurb=environment_blurb
    )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
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
