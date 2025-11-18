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

from tools import create_tools, initialize_vector_store_manager, initialize_car, initialize_robot_vision_tools


def create_agent():
    """Create and configure the agent with tools and memory."""
    
    # Initialize vector store manager
    print("Initializing vector stores...")
    initialize_vector_store_manager()
    
    # Initialize robot car
    initialize_car()
    
    # Create the LLM
    llm = ChatOpenAI(
        model="gpt-5",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize robot vision tools with the LLM
    initialize_robot_vision_tools(llm)
    
    # Create tools
    tools = create_tools()
    
    # Load the prompt from the external file
    prompt_file_path = os.path.join(os.path.dirname(__file__), "prompts", "global_prompt.txt")
    with open(prompt_file_path, 'r') as f:
        prompt_text = f.read()
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text.format(cwd=os.getcwd())),
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
        max_iterations=15,
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

