"""
Main entry point for the Experience Engine agent.
Handles the user interaction loop.
"""

from dotenv import load_dotenv

from agent import create_conversational_agent


# Load environment variables
load_dotenv()


def main():
    """Main function to run the agent."""
    
    print("=" * 60)
    print("Experience Engine Agent")
    print("=" * 60)
    print("\nInitializing agent...")
    
    # Create agent with memory
    agent, message_history = create_conversational_agent()
    
    print("\nAgent initialized! Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear' to clear conversation history.")
    print("=" * 60)
    print()
    
    session_id = "default_session"
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
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
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")
            print("-" * 60)
            print()


if __name__ == "__main__":
    main()

