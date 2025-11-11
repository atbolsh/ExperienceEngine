#!/usr/bin/env python3
"""
Example script demonstrating the scripting capabilities.
This can be written and executed by the agent.
"""

import sys

def main():
    print("Hello from an agent-created script!")
    print(f"Python version: {sys.version}")
    
    if len(sys.argv) > 1:
        print(f"\nReceived arguments: {sys.argv[1:]}")
        for i, arg in enumerate(sys.argv[1:], 1):
            print(f"  Arg {i}: {arg}")
    else:
        print("\nNo arguments provided.")
    
    print("\nScript completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

