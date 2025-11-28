#!/usr/bin/env python3
"""
Demo script to show what continuity information the agent will see at startup.
This demonstrates the new session continuity features.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import load_context_prompt, load_most_recent_episodic_memory

def demo_continuity():
    print("=" * 70)
    print("SESSION CONTINUITY DEMO")
    print("=" * 70)
    print()
    print("This shows what the agent will automatically see at startup.")
    print()
    
    # Load context prompt
    print("=" * 70)
    print("1. LEARNED CONTEXT (from context_prompt.md)")
    print("=" * 70)
    context = load_context_prompt()
    if context:
        print(context)
    else:
        print("(No context hints found)")
    print()
    
    # Load most recent episodic memory
    print("=" * 70)
    print("2. SESSION CONTINUITY (most recent episodic memory)")
    print("=" * 70)
    recent_memory = load_most_recent_episodic_memory()
    if recent_memory:
        print(recent_memory)
    else:
        print("(No episodic memories found)")
    print()
    
    print("=" * 70)
    print("END OF CONTINUITY DEMO")
    print("=" * 70)
    print()
    print("The agent will see the above information automatically appended")
    print("to its system prompt on every startup, providing continuity")
    print("between sessions without manual intervention.")
    print()

if __name__ == "__main__":
    demo_continuity()

