"""
Memory writing tools.
Provides tools for creating new memories in semantic, procedural, and episodic directories.
"""

import os
from pathlib import Path
from typing import List

from langchain_compat import BaseModel, Field, StructuredTool, Tool


# Memory directories
SEMANTIC_DIR = Path("semantic")
PROCEDURAL_DIR = Path("procedural")
EPISODIC_DIR = Path("episodic")
WORKING_DIR = Path("working")


# ===== Input Schemas =====

class WriteMemoryInput(BaseModel):
    """Input for memory writing tools."""
    memory_name: str = Field(description="Name of the memory folder to create. Should NOT contain special characters; use underscores (_) instead of spaces")
    content: str = Field(description="Text content to write to the memory's guide.txt file")


# ===== Tool Functions =====

def write_semantic_memory(memory_name: str, content: str) -> str:
    """
    Create a new semantic memory with the given name and content.
    Semantic memories store descriptive information, concepts, and definitions.
    
    Args:
        memory_name: Name of the memory folder to create
        content: Text content to write to guide.txt
        
    Returns:
        Success or error message
    """
    try:
        # Security: prevent path traversal
        if '..' in memory_name or '/' in memory_name or '\\' in memory_name:
            return "Error: Invalid memory name. Memory name cannot contain path separators or '..'."
        
        # Create memory directory
        memory_path = SEMANTIC_DIR / memory_name
        memory_path.mkdir(parents=True, exist_ok=True)
        
        # Write content to guide.txt
        guide_path = memory_path / 'guide.txt'
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully created semantic memory '{memory_name}' with {len(content)} characters."
    
    except Exception as e:
        return f"Error creating semantic memory: {str(e)}"


def write_procedural_memory(memory_name: str, content: str) -> str:
    """
    Create a new procedural memory with the given name and content.
    Procedural memories store step-by-step instructions and how-to guides.
    
    Args:
        memory_name: Name of the memory folder to create
        content: Text content to write to guide.txt
        
    Returns:
        Success or error message
    """
    try:
        # Security: prevent path traversal
        if '..' in memory_name or '/' in memory_name or '\\' in memory_name:
            return "Error: Invalid memory name. Memory name cannot contain path separators or '..'."
        
        # Create memory directory
        memory_path = PROCEDURAL_DIR / memory_name
        memory_path.mkdir(parents=True, exist_ok=True)
        
        # Write content to guide.txt
        guide_path = memory_path / 'guide.txt'
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully created procedural memory '{memory_name}' with {len(content)} characters."
    
    except Exception as e:
        return f"Error creating procedural memory: {str(e)}"


def write_episodic_memory(memory_name: str, content: str) -> str:
    """
    Create a new episodic memory with the given name and content.
    Episodic memories store detailed records of past interactions and experiences.
    
    Args:
        memory_name: Name of the memory folder to create
        content: Text content to write to guide.txt
        
    Returns:
        Success or error message
    """
    try:
        # Security: prevent path traversal
        if '..' in memory_name or '/' in memory_name or '\\' in memory_name:
            return "Error: Invalid memory name. Memory name cannot contain path separators or '..'."
        
        # Create memory directory
        memory_path = EPISODIC_DIR / memory_name
        memory_path.mkdir(parents=True, exist_ok=True)
        
        # Write content to guide.txt
        guide_path = memory_path / 'guide.txt'
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully created episodic memory '{memory_name}' with {len(content)} characters."
    
    except Exception as e:
        return f"Error creating episodic memory: {str(e)}"


def write_working_memory(memory_name: str, content: str) -> str:
    """
    Create a new working memory with the given name and content.
    Working memories store temporary observations and recent captures for the current session.
    
    IMPORTANT: Working memory is automatically cleared at the end of each session.
    Store important findings in semantic/procedural/episodic memory for long-term retention.
    
    Args:
        memory_name: Name of the memory folder to create
        content: Text content to write to guide.txt
        
    Returns:
        Success or error message
    """
    try:
        # Security: prevent path traversal
        if '..' in memory_name or '/' in memory_name or '\\' in memory_name:
            return "Error: Invalid memory name. Memory name cannot contain path separators or '..'."
        
        # Create memory directory
        memory_path = WORKING_DIR / memory_name
        memory_path.mkdir(parents=True, exist_ok=True)
        
        # Write content to guide.txt
        guide_path = memory_path / 'guide.txt'
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully created working memory '{memory_name}' with {len(content)} characters. NOTE: Working memory is cleared at session end."
    
    except Exception as e:
        return f"Error creating working memory: {str(e)}"


# ===== Tool Definitions =====

def create_memory_writing_tools() -> List[Tool]:
    """Create and return memory writing tools."""
    return [
        StructuredTool(
            name="write_semantic_memory",
            func=write_semantic_memory,
            description="Create a new semantic memory for storing descriptive information, concepts, and definitions. Use when you need to remember WHAT something is. IMPORTANT: Memory names should NOT contain special characters; use underscores (_) instead of spaces. After creating, use refresh_vector_stores to make it searchable. Inputs: memory_name (string, name for the memory folder), content (string, text content for guide.txt)",
            args_schema=WriteMemoryInput,
        ),
        StructuredTool(
            name="write_procedural_memory",
            func=write_procedural_memory,
            description="Create a new procedural memory for storing step-by-step instructions and how-to guides. Use when you need to remember HOW to do something. IMPORTANT: Memory names should NOT contain special characters; use underscores (_) instead of spaces. After creating, use refresh_vector_stores to make it searchable. Inputs: memory_name (string, name for the memory folder), content (string, text content for guide.txt)",
            args_schema=WriteMemoryInput,
        ),
        StructuredTool(
            name="write_episodic_memory",
            func=write_episodic_memory,
            description="Create a new episodic memory for storing detailed records of past interactions and experiences. Use when you need to remember specific events. IMPORTANT: Memory names should NOT contain special characters; use underscores (_) instead of spaces. After creating, use refresh_vector_stores to make it searchable. Inputs: memory_name (string, name for the memory folder), content (string, text content for guide.txt)",
            args_schema=WriteMemoryInput,
        ),
        StructuredTool(
            name="write_working_memory",
            func=write_working_memory,
            description="Create a new working memory for temporary session observations like recent camera captures and notes. CLEARED AT SESSION END - use for temporary storage only. For important info, use semantic/procedural/episodic memory. Memory names should NOT contain special characters; use underscores (_) instead of spaces. After creating, use refresh_vector_stores to make it searchable. Inputs: memory_name (string, name for the memory folder), content (string, text content for guide.txt)",
            args_schema=WriteMemoryInput,
        ),
    ]

