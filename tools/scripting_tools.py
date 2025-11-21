"""
Scripting tools.
Provides tools for writing and executing Python scripts in the scripts/ directory.
"""

import os
import subprocess
from pathlib import Path
from typing import List

from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field


# Default scripts directory
SCRIPTS_DIR = Path("scripts")


# ===== Input Schemas =====

class WriteScriptInput(BaseModel):
    """Input for write_script tool."""
    script_name: str = Field(description="Name of the script file (e.g., 'my_script.py')")
    script_content: str = Field(description="Complete Python code to write to the script")


class ExecuteScriptInput(BaseModel):
    """Input for execute_script tool."""
    script_name: str = Field(description="Name of the script file to execute (e.g., 'my_script.py')")
    args: str = Field(default="", description="Command-line arguments to pass to the script (optional)")


class ListScriptsInput(BaseModel):
    """Input for list_scripts tool."""
    pass


class ReadScriptInput(BaseModel):
    """Input for read_script tool."""
    script_name: str = Field(description="Name of the script file to read")


class DeleteScriptInput(BaseModel):
    """Input for delete_script tool."""
    script_name: str = Field(description="Name of the script file to delete")


# ===== Tool Functions =====

def write_script(script_name: str, script_content: str) -> str:
    """
    Write a new Python script to the scripts/ directory.
    
    Args:
        script_name: Name of the script file (should end with .py)
        script_content: Complete Python code content
        
    Returns:
        Success or error message
    """
    try:
        # Ensure scripts directory exists
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Ensure the script name ends with .py
        if not script_name.endswith('.py'):
            script_name += '.py'
        
        # Security: prevent path traversal
        if '..' in script_name or '/' in script_name or '\\' in script_name:
            return "Error: Invalid script name. Script name cannot contain path separators or '..'."
        
        script_path = SCRIPTS_DIR / script_name
        
        # Write the script
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make the script executable (Unix-like systems)
        try:
            os.chmod(script_path, 0o755)
        except Exception:
            pass  # On Windows, this might fail, but that's okay
        
        return f"Successfully wrote script '{script_name}' to scripts/ directory ({len(script_content)} characters)."
    
    except Exception as e:
        return f"Error writing script: {str(e)}"


def execute_script(script_name: str, args: str = "") -> str:
    """
    Execute a Python script from the scripts/ directory.
    
    Args:
        script_name: Name of the script file to execute
        args: Command-line arguments to pass to the script
        
    Returns:
        Script output or error message
    """
    try:
        # Ensure the script name ends with .py
        if not script_name.endswith('.py'):
            script_name += '.py'
        
        # Security: prevent path traversal
        if '..' in script_name or '/' in script_name or '\\' in script_name:
            return "Error: Invalid script name. Script name cannot contain path separators or '..'."
        
        script_path = SCRIPTS_DIR / script_name
        
        # Check if script exists
        if not script_path.exists():
            return f"Error: Script '{script_name}' not found in scripts/ directory."
        
        # Prepare command
        cmd = ['python', str(script_path)]
        if args:
            # Split args by spaces (simple splitting - may need improvement for quoted args)
            cmd.extend(args.split())
        
        # Execute the script with a timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            cwd=os.getcwd()
        )
        
        # Format output
        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        output.append(f"\nReturn code: {result.returncode}")
        
        if result.returncode == 0:
            return f"Script '{script_name}' executed successfully.\n\n" + "\n".join(output)
        else:
            return f"Script '{script_name}' execution failed.\n\n" + "\n".join(output)
    
    except subprocess.TimeoutExpired:
        return f"Error: Script '{script_name}' execution timed out after 30 seconds."
    except Exception as e:
        return f"Error executing script: {str(e)}"


def list_scripts(dummy_input: str = "") -> str:
    """
    List all Python scripts in the scripts/ directory.
    
    Args:
        dummy_input: Unused parameter (for LangChain Tool compatibility)
    
    Returns:
        List of available scripts
    """
    try:
        # Ensure scripts directory exists
        if not SCRIPTS_DIR.exists():
            return "Scripts directory is empty (directory doesn't exist yet)."
        
        # Get all .py files
        scripts = sorted(SCRIPTS_DIR.glob("*.py"))
        
        if not scripts:
            return "No Python scripts found in scripts/ directory."
        
        output = ["Available scripts in scripts/ directory:"]
        for script in scripts:
            size = script.stat().st_size
            output.append(f"  - {script.name} ({size} bytes)")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Error listing scripts: {str(e)}"


def read_script(script_name: str) -> str:
    """
    Read the contents of a script from the scripts/ directory.
    
    Args:
        script_name: Name of the script file to read
        
    Returns:
        Script contents or error message
    """
    try:
        # Ensure the script name ends with .py
        if not script_name.endswith('.py'):
            script_name += '.py'
        
        # Security: prevent path traversal
        if '..' in script_name or '/' in script_name or '\\' in script_name:
            return "Error: Invalid script name. Script name cannot contain path separators or '..'."
        
        script_path = SCRIPTS_DIR / script_name
        
        # Check if script exists
        if not script_path.exists():
            return f"Error: Script '{script_name}' not found in scripts/ directory."
        
        # Read the script
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"Contents of '{script_name}':\n\n{content}"
    
    except Exception as e:
        return f"Error reading script: {str(e)}"


def delete_script(script_name: str) -> str:
    """
    Delete a script from the scripts/ directory.
    
    Args:
        script_name: Name of the script file to delete
        
    Returns:
        Success or error message
    """
    try:
        # Ensure the script name ends with .py
        if not script_name.endswith('.py'):
            script_name += '.py'
        
        # Security: prevent path traversal
        if '..' in script_name or '/' in script_name or '\\' in script_name:
            return "Error: Invalid script name. Script name cannot contain path separators or '..'."
        
        script_path = SCRIPTS_DIR / script_name
        
        # Check if script exists
        if not script_path.exists():
            return f"Error: Script '{script_name}' not found in scripts/ directory."
        
        # Delete the script
        script_path.unlink()
        
        return f"Successfully deleted script '{script_name}' from scripts/ directory."
    
    except Exception as e:
        return f"Error deleting script: {str(e)}"


# ===== Tool Definitions =====

def create_scripting_tools() -> List[Tool]:
    """Create and return scripting tools."""
    return [
        StructuredTool(
            name="write_script",
            func=write_script,
            description="Write a new Python script to the scripts/ directory. Inputs: script_name (string, e.g., 'my_script.py'), script_content (string, complete Python code)",
            args_schema=WriteScriptInput,
        ),
        StructuredTool(
            name="execute_script",
            func=execute_script,
            description="Execute a Python script from the scripts/ directory. Inputs: script_name (string, e.g., 'my_script.py'), args (optional string, command-line arguments)",
            args_schema=ExecuteScriptInput,
        ),
        Tool(
            name="list_scripts",
            func=list_scripts,
            description="List all Python scripts available in the scripts/ directory. No input required.",
        ),
        StructuredTool(
            name="read_script",
            func=read_script,
            description="Read the contents of a script from the scripts/ directory. Input: script_name (string, e.g., 'my_script.py')",
            args_schema=ReadScriptInput,
        ),
        StructuredTool(
            name="delete_script",
            func=delete_script,
            description="Delete a script from the scripts/ directory. Input: script_name (string, e.g., 'my_script.py')",
            args_schema=DeleteScriptInput,
        ),
    ]

