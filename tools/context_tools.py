"""
Context prompt editing tool.
Allows the agent to update short-term context hints stored in prompts/context_prompt.md.
"""

import os
from typing import List
from pathlib import Path

from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field, validator


class EditContextPromptInput(BaseModel):
    """Input for editing the context prompt."""
    new_content: str = Field(
        description="The new content for context_prompt.md. Must be concise (max 1000 chars or 30 lines). Consider tradeoffs between keeping valuable old information and adding new insights."
    )
    
    @validator('new_content')
    def validate_content(cls, v):
        """Validate content length constraints."""
        lines = v.split('\n')
        if len(lines) > 30:
            raise ValueError(f"Content exceeds 30 line limit (has {len(lines)} lines)")
        if len(v) > 1000:
            raise ValueError(f"Content exceeds 1000 character limit (has {len(v)} chars)")
        return v


def edit_context_prompt(new_content: str) -> str:
    """
    Edit the context_prompt.md file with new learned hints.
    
    This file stores useful hints (max 30 lines / 1000 chars) about the 
    environment that the agent has learned. These hints persist across sessions.
    
    IMPORTANT: When editing, you must carefully consider the tradeoff between:
    - Keeping valuable old information that remains relevant
    - Adding new insights that could improve future performance
    
    The hard limits are:
    - Maximum 30 lines
    - Maximum 1000 characters
    
    Args:
        new_content: The complete new content for context_prompt.md
        
    Returns:
        Success or error message
    """
    try:
        # Validate constraints
        lines = new_content.split('\n')
        if len(lines) > 30:
            return f"Error: Content has {len(lines)} lines (max 30 allowed). Please condense."
        
        if len(new_content) > 1000:
            return f"Error: Content has {len(new_content)} characters (max 1000 allowed). Please condense."
        
        # Get path to context_prompt.md
        context_path = Path(__file__).parent.parent / "prompts" / "context_prompt.md"
        
        # Read old content for reference
        old_content = ""
        if context_path.exists():
            with open(context_path, 'r', encoding='utf-8') as f:
                old_content = f.read()
        
        # Write new content
        with open(context_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return f"Successfully updated context_prompt.md ({len(new_content)} chars, {len(lines)} lines).\n\nOld content was:\n{old_content}\n\nNew content is:\n{new_content}"
        
    except Exception as e:
        return f"Error updating context_prompt.md: {str(e)}"


def read_context_prompt() -> str:
    """
    Read the current contents of context_prompt.md.
    
    Returns:
        Current content or error message
    """
    try:
        context_path = Path(__file__).parent.parent / "prompts" / "context_prompt.md"
        
        if not context_path.exists():
            return "Error: context_prompt.md does not exist yet."
        
        with open(context_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        return f"Current context_prompt.md ({len(content)} chars, {len(lines)} lines):\n\n{content}"
        
    except Exception as e:
        return f"Error reading context_prompt.md: {str(e)}"


def create_context_tools() -> List[Tool]:
    """Create and return context prompt tools."""
    return [
        StructuredTool(
            name="edit_context_prompt",
            func=edit_context_prompt,
            description="Edit the context_prompt.md file to update learned hints. Max 1000 chars OR 30 lines. You must balance keeping valuable old info with adding new insights. Input: new_content (the complete new content)",
            args_schema=EditContextPromptInput,
        ),
        Tool(
            name="read_context_prompt",
            func=read_context_prompt,
            description="Read the current contents of context_prompt.md to see what hints are currently stored.",
        ),
    ]

