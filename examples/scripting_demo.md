# Scripting Tools Demo

The agent has powerful scripting capabilities through the `scripting_tools.py` module.

## Available Tools

### 1. `write_script`
Creates a new Python script in the `scripts/` directory.

**Example:**
```
Agent: I'll create a script for you.
[Uses write_script tool with script_name="hello.py" and script_content="print('Hello!')"]
```

### 2. `execute_script`
Executes a script from the `scripts/` directory with optional arguments.

**Example:**
```
Agent: I'll run that script now.
[Uses execute_script tool with script_name="hello.py" and args=""]
```

**With arguments:**
```
[Uses execute_script tool with script_name="process_data.py" and args="input.txt output.txt"]
```

### 3. `list_scripts`
Lists all Python scripts in the `scripts/` directory.

**Example:**
```
Agent: Let me see what scripts are available.
[Uses list_scripts tool]
```

### 4. `read_script`
Reads the contents of a script.

**Example:**
```
Agent: Let me check what's in that script.
[Uses read_script tool with script_name="hello.py"]
```

### 5. `delete_script`
Removes a script from the `scripts/` directory.

**Example:**
```
Agent: I'll delete that old script.
[Uses delete_script tool with script_name="old_script.py"]
```

## Security Features

1. **Path Traversal Prevention**: Scripts must be in the `scripts/` directory
   - Blocks attempts to use `..`, `/`, or `\` in filenames
   
2. **Execution Timeout**: Scripts have a 30-second execution limit
   - Prevents runaway processes
   
3. **Isolated Directory**: All scripts are confined to `scripts/`
   - Cannot accidentally overwrite system files
   
4. **Automatic Extension**: `.py` extension added if not provided
   - Ensures only Python scripts are created

## Example Conversation

**User:** "Create a script that prints the current date and time"

**Agent:**
1. Uses `write_script` to create `date_time.py`:
```python
from datetime import datetime
print(f"Current date and time: {datetime.now()}")
```

2. Uses `execute_script` to run it:
```
Script 'date_time.py' executed successfully.

STDOUT:
Current date and time: 2025-11-11 11:30:45.123456

Return code: 0
```

**User:** "List all my scripts"

**Agent:** Uses `list_scripts`:
```
Available scripts in scripts/ directory:
  - date_time.py (78 bytes)
  - hello.py (21 bytes)
```

## Use Cases

- **Automation**: Create scripts for repetitive tasks
- **Data Processing**: Write scripts to transform or analyze data
- **Testing**: Create test scripts and run them
- **Prototyping**: Quick Python code experiments
- **Utilities**: Build helper scripts for common operations

