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

from tools import create_tools, initialize_vector_store_manager


def create_agent():
    """Create and configure the agent with tools and memory."""
    
    # Initialize vector store manager
    print("Initializing vector stores...")
    initialize_vector_store_manager()
    
    # Create the LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create tools
    tools = create_tools()
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to multiple capabilities:

1. **Memory Systems**: You have access to three types of long-term memory:
   - Semantic Memory: Descriptive information, concepts, and definitions
   - Procedural Memory: Step-by-step instructions and how-to guides  
   - Episodic Memory: Detailed records of past interactions and experiences
   
   Memories are stored as folders containing text (always indexed) and optionally images (0-10 per memory).
   Text is searchable via RAG tools. Images can be accessed and analyzed on-demand using image tools.
   Use these memory systems ONLY when relevant to the user's query. Don't query them unnecessarily.

2. **Filesystem Operations**: You can list directories, read/write files, create directories, delete files, etc.

3. **Scripting Capabilities**: You can write, execute, read, list, and delete Python scripts in the scripts/ directory. This allows you to create reusable automation scripts and execute complex tasks.

4. **Sequential Thinking**: For complex problems requiring step-by-step reasoning, you can use the sequential_thinking_tool to break down your thought process into clear, logical steps.

5. **Conversation Memory**: You maintain context across the conversation and can reference previous exchanges.

Guidelines:
- Be helpful, accurate, and concise
- Only query the memory systems when the information would be genuinely useful
- When using filesystem tools, provide clear feedback about operations
- Scripts are stored in the scripts/ directory and execute with a 30-second timeout
- For complex multi-step problems, consider using the sequential thinking tool to organize your reasoning
- If you're unsure about something, say so rather than making assumptions
- After adding new documents to the semantic/procedural/episodic folders, use refresh_vector_stores to update the indices

Current working directory: {cwd}
""".format(cwd=os.getcwd())),
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

