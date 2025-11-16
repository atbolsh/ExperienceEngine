"""
Main entry point for the Experience Engine agent.
Handles the user interaction loop.
"""

import os
from dotenv import load_dotenv

from agent import create_conversational_agent
from tools.session_tools import get_session_control_signal, reset_session_control_signal


# Load environment variables
load_dotenv()


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
            {"input": closing_prompt},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"Assistant: {response['output']}\n")
        print(f"{'=' * 60}\n")
    except Exception as e:
        print(f"Error during closing sequence: {str(e)}\n")


def main():
    """Main function to run the agent."""
    
    session_id = "default_session"
    
    while True:  # Outer loop for restart functionality
        print("=" * 60)
        print("Experience Engine Agent")
        print("=" * 60)
        print("\nInitializing agent...")
        
        # Create agent with memory
        agent, message_history = create_conversational_agent()
        
        print("\nAgent initialized! Type 'exit' or 'quit' to end the conversation.")
        print("Type 'clear' to clear conversation history.")
        print("The agent can also choose to end or restart the session using its tools.")
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
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    run_closing_sequence(agent, session_id, "User ended the session")
                    print("Goodbye!")
                    return  # Exit the program completely
                
                # Check for clear command
                if user_input.lower() == 'clear':
                    message_history.clear()
                    print("\n[Conversation history cleared]\n")
                    continue
                
                # Run the agent
                print()
                response = agent.invoke(
                    {"input": user_input},
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

