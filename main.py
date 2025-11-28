"""
Main entry point for the Experience Engine agent.
Handles the user interaction loop and loads environment dynamically.
"""

import os
from dotenv import load_dotenv

from agent import create_conversational_agent
from tools.session_tools import get_session_control_signal, reset_session_control_signal
from tools import clear_working_memory_function, close_env

# Load environment variables
load_dotenv()


def load_environment_inject_image():
    """
    Dynamically load the inject_image function from the active environment.
    
    Returns:
        The inject_image function from the active environment
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
    
    try:
        if env_name == 'car_environment':
            from environments.car_environment import inject_image
            return inject_image
        elif env_name == 'game_environment':
            from environments.game_environment import inject_image
            return inject_image
        else:
            print(f"Unknown environment '{env_name}', defaulting to car_environment")
            from environments.car_environment import inject_image
            return inject_image
    except ImportError as e:
        print(f"Error importing environment: {e}")
        from environments.car_environment import inject_image
        return inject_image


# Load the inject_image function from active environment
inject_image = load_environment_inject_image()


def load_closing_prompt() -> str:
    """Load the closing prompt from file."""
    prompt_file_path = os.path.join(os.path.dirname(__file__), "prompts", "closing_prompt.txt")
    with open(prompt_file_path, 'r') as f:
        return f.read()


def run_closing_sequence(agent, session_id: str, reason: str = "User ended the session"):
    """Run the closing sequence with the agent."""
    print(f"\n{'=' * 60}")
    print(f"Closing session: {reason}")
    print(f"{'=' * 60}\n")
    
    try:
        closing_prompt = load_closing_prompt()
        print("Agent: Reflecting on the session...\n")
        
        response = agent.invoke(
            {"input": inject_image(closing_prompt)},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"Assistant: {response['output']}\n")
        print(f"{'=' * 60}\n")
    except Exception as e:
        print(f"Error during closing sequence: {str(e)}\n")
    
    # Clear working memory at the end of each session
    try:
        print("Clearing working memory...")
        clear_working_memory_function()
        print("Working memory cleared.\n")
    except Exception as e:
        print(f"Error clearing working memory: {str(e)}\n")
    
    # Close the environment connection
    close_env()


def main():
    """Main function to run the agent."""
    session_id = "default_session"
    
    while True:  # Outer loop for restart functionality
        print("=" * 60)
        print("Experience Engine - Autonomous Exploration Agent")
        print("=" * 60)
        print("\nInitializing exploration agent...")
        
        # Create agent with memory
        agent, message_history = create_conversational_agent()
        
        print("\nAgent ready to explore!")
        print("\nThis agent is designed to autonomously explore its environment,")
        print("discover new information, and record insights in its memory systems.")
        print("You can guide and give hints, but the agent drives the exploration.")
        print("\nCommands:")
        print("  'exit'/'quit' - End session with closing sequence")
        print("  'EXIT' (all caps) - Abort immediately without closing sequence")
        print("  'clear' - Reset conversation history")
        print("The agent can also end or restart sessions using its tools.")
        print("=" * 60)
        print()
        
        restart_requested = False
        
        # Inner loop for conversation
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for immediate abort (EXIT in all caps)
                if user_input == 'EXIT':
                    print("\n[Aborting session immediately without closing sequence]")
                    close_env()
                    print("Goodbye!")
                    return  # Exit the program completely
                
                # Check for exit commands with closing sequence
                if user_input.lower() in ['exit', 'quit', 'q']:
                    run_closing_sequence(agent, session_id, "User ended the session")
                    print("Goodbye!")
                    return  # Exit the program completely
                
                # Check for clear command
                if user_input.lower() == 'clear':
                    message_history.clear()
                    print("\n[Conversation history cleared]\n")
                    continue
                
                # Run the agent with image injection
                print()
                response = agent.invoke(
                    {"input": inject_image(user_input)},
                    config={"configurable": {"session_id": session_id}}
                )
                
                # Print the response
                print(f"\nAssistant: {response['output']}\n")
                print("-" * 60)
                print()
                
                # Check for session control signals from agent
                signal = get_session_control_signal()
                if signal["action"] == "end":
                    reset_session_control_signal()
                    run_closing_sequence(agent, session_id, "Agent ended the session")
                    print("Goodbye!")
                    return  # Exit the program completely
                elif signal["action"] == "restart":
                    reset_session_control_signal()
                    run_closing_sequence(agent, session_id, "Agent restarting the session")
                    restart_requested = True
                    break  # Break inner loop to restart
                
            except KeyboardInterrupt:
                print("\n")
                run_closing_sequence(agent, session_id, "User interrupted (Ctrl+C)")
                print("Goodbye!")
                return  # Exit the program completely
            except Exception as e:
                print(f"\nError: {str(e)}\n")
                print("-" * 60)
                print()
        
        # If we broke out of inner loop and restart wasn't requested, exit completely
        if not restart_requested:
            break
        
        # Otherwise, continue to restart (outer loop continues)


if __name__ == "__main__":
    main()
