"""
Filesystem manipulation tools.
Provides tools for file and directory operations.
"""

import shutil
from pathlib import Path
from typing import List

from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field


# ===== Input Schemas =====

class ListDirectoryInput(BaseModel):
    """Input for list_directory tool."""
    path: str = Field(default=".", description="Directory path to list (default: current directory)")


class ReadFileInput(BaseModel):
    """Input for read_file tool."""
    file_path: str = Field(description="Path to the file to read")


class WriteFileInput(BaseModel):
    """Input for write_file tool."""
    file_path: str = Field(description="Path to the file to write. File names should not contain special characters; use underscores instead of spaces")
    content: str = Field(description="Content to write to the file")


class AppendFileInput(BaseModel):
    """Input for append_file tool."""
    file_path: str = Field(description="Path to the file to append to")
    content: str = Field(description="Content to append to the file")


class DeleteFileInput(BaseModel):
    """Input for delete_file tool."""
    file_path: str = Field(description="Path to the file to delete")


class CreateDirectoryInput(BaseModel):
    """Input for create_directory tool."""
    dir_path: str = Field(description="Path to the directory to create. Directory names should not contain special characters; use underscores instead of spaces")


# ===== Tool Functions =====

def list_directory(path: str = ".") -> str:
    """List files and directories in the specified path."""
    try:
        target_path = Path(path).resolve()
        if not target_path.exists():
            return f"Error: Path '{path}' does not exist."
        if not target_path.is_dir():
            return f"Error: Path '{path}' is not a directory."
        
        items = []
        for item in sorted(target_path.iterdir()):
            if item.is_dir():
                items.append(f"[DIR]  {item.name}/")
            else:
                size = item.stat().st_size
                items.append(f"[FILE] {item.name} ({size} bytes)")
        
        if not items:
            return f"Directory '{path}' is empty."
        
        return f"Contents of '{path}':\n" + "\n".join(items)
    except Exception as e:
        return f"Error listing directory: {str(e)}"


def read_file(file_path: str) -> str:
    """Read and return the contents of a file."""
    try:
        target_path = Path(file_path).resolve()
        if not target_path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not target_path.is_file():
            return f"Error: Path '{file_path}' is not a file."
        
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"Contents of '{file_path}':\n\n{content}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating it if it doesn't exist."""
    try:
        target_path = Path(file_path).resolve()
        
        # Create parent directories if they don't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote {len(content)} characters to '{file_path}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"


def append_file(file_path: str, content: str) -> str:
    """Append content to an existing file."""
    try:
        target_path = Path(file_path).resolve()
        
        # Create parent directories if they don't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_path, 'a', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully appended {len(content)} characters to '{file_path}'."
    except Exception as e:
        return f"Error appending to file: {str(e)}"


def get_current_directory(dummy_input: str = "") -> str:
    """
    Get the current working directory.
    
    Args:
        dummy_input: Unused parameter (for LangChain Tool compatibility)
    """
    try:
        cwd = Path.cwd()
        return f"Current directory: {cwd}"
    except Exception as e:
        return f"Error getting current directory: {str(e)}"


def delete_file(file_path: str) -> str:
    """Delete a file or directory."""
    try:
        target_path = Path(file_path).resolve()
        if not target_path.exists():
            return f"Error: Path '{file_path}' does not exist."
        
        if target_path.is_file():
            target_path.unlink()
            return f"Successfully deleted file '{file_path}'."
        elif target_path.is_dir():
            shutil.rmtree(target_path)
            return f"Successfully deleted directory '{file_path}' and all its contents."
        else:
            return f"Error: Path '{file_path}' is neither a file nor a directory."
    except Exception as e:
        return f"Error deleting path: {str(e)}"


def create_directory(dir_path: str) -> str:
    """Create a new directory."""
    try:
        target_path = Path(dir_path).resolve()
        target_path.mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory '{dir_path}'."
    except Exception as e:
        return f"Error creating directory: {str(e)}"


# ===== Tool Definitions =====

def create_filesystem_tools() -> List[Tool]:
    """Create and return filesystem tools."""
    return [
        StructuredTool(
            name="list_directory",
            func=list_directory,
            description="List files and directories in a specified path. Input: directory path (default: current directory)",
            args_schema=ListDirectoryInput,
        ),
        StructuredTool(
            name="read_file",
            func=read_file,
            description="Read and return the contents of a file. Input: file path",
            args_schema=ReadFileInput,
        ),
        StructuredTool(
            name="write_file",
            func=write_file,
            description="Write content to a file, creating it if it doesn't exist. IMPORTANT: File names should NOT contain special characters; use underscores (_) instead of spaces. Inputs: file path, content",
            args_schema=WriteFileInput,
        ),
        StructuredTool(
            name="append_file",
            func=append_file,
            description="Append content to an existing file. Inputs: file path, content",
            args_schema=AppendFileInput,
        ),
        Tool(
            name="get_current_directory",
            func=get_current_directory,
            description="Get the current working directory. No input required.",
        ),
        StructuredTool(
            name="delete_file",
            func=delete_file,
            description="Delete a file or directory (including all contents if directory). Input: file or directory path",
            args_schema=DeleteFileInput,
        ),
        StructuredTool(
            name="create_directory",
            func=create_directory,
            description="Create a new directory. IMPORTANT: Directory names should NOT contain special characters; use underscores (_) instead of spaces. Input: directory path",
            args_schema=CreateDirectoryInput,
        ),
    ]

