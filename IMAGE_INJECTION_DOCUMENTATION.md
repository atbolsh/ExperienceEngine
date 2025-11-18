# Image Injection Implementation

## Overview

This document describes the implementation of automatic image injection into all LLM messages. Every message sent to the LLM now includes an image - either from the robot camera capture or from working memory.

## Changes Made

### 1. Custom LLM Wrapper (`utils/llm_wrapper.py`)

Created a new `ImageInjectingLLM` class that extends `ChatOpenAI` and automatically injects images into all messages.

**Key Features:**
- Inherits from `ChatOpenAI` to maintain full compatibility with LangChain
- Overrides `invoke()`, `ainvoke()`, `_generate()`, and `_agenerate()` methods
- Injects images into the last `HumanMessage` in every message chain
- Supports both string and multi-modal message formats
- Prevents duplicate image injection if an image is already present

**Image Source Priority:**
1. **Primary**: Latest robot camera capture (via `get_latest_robot_image()`)
2. **Fallback**: Most recent image file from working memory directory

**Image Format:**
Images are encoded as base64 and injected using OpenAI's vision format:
```python
{
    "type": "image_url",
    "image_url": {
        "url": f"data:image/{format};base64,{base64_data}"
    }
}
```

### 2. Image Utility Functions (`utils/image_utils.py`)

Created a new utility module for handling image operations with unique timestamps.

**Functions:**
- `generate_unique_image_filename(prefix, extension)` - Generates timestamp-based unique filenames
- `save_image_with_unique_name(image_array, directory, ...)` - Saves images with automatic unique naming
- `get_unique_filepath(directory, filename)` - Returns unique filepath, adding timestamp if file exists

**Timestamp Format:**
- Pattern: `{prefix}_{YYYYMMDD_HHMMSS_mmm}.{extension}`
- Example: `capture_20251118_143027_456.jpg`
- Includes milliseconds for uniqueness even with rapid captures

### 3. Updated Robot Tools (`tools/robot_tools.py`)

Modified `capture_robot_image()` to save images with unique timestamps:

**Changes:**
- Added `datetime` import for timestamp generation
- Images now saved with format: `capture_{timestamp}.jpg`
- Maintains backward compatibility by also saving as `latest_capture.jpg`
- Updated return message to include both filenames

**Example Output:**
```
Successfully captured image from robot camera. 
Image saved to working/capture_20251118_143027_456.jpg and working/latest_capture.jpg. 
The image is now available for analysis.
```

### 4. Updated Agent Configuration (`agent.py`)

Changed the LLM initialization to use the custom wrapper:

**Before:**
```python
llm = ChatOpenAI(
    model="gpt-5",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)
```

**After:**
```python
from utils.llm_wrapper import ImageInjectingLLM

llm = ImageInjectingLLM(
    model="gpt-5",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## How It Works

### Normal Flow (With Robot Image)

1. User sends a message to the agent
2. Agent processes the message and prepares to call the LLM
3. `ImageInjectingLLM` intercepts the call
4. It retrieves the latest robot image via `get_latest_robot_image()`
5. Image is encoded to base64
6. Image is injected into the last `HumanMessage`
7. Modified message is sent to OpenAI's API
8. Response is returned normally

### Fallback Flow (Without Robot Image)

1. If no robot image is available (`get_latest_robot_image()` returns `None`)
2. Wrapper checks the `working/` directory for image files
3. Finds the most recently modified image file
4. Uses that image instead
5. Continues with normal flow

### Edge Cases

- **No Images Available**: If neither robot image nor working memory images exist, messages are sent without images
- **Already Has Image**: If a message already contains an image, no duplicate is added
- **Multi-modal Messages**: Existing multi-modal content is preserved and extended

## Benefits

1. **Automatic Context**: Every LLM interaction has visual context
2. **No Code Changes Required**: Existing tools work without modification
3. **Transparent**: Works invisibly in the background
4. **Flexible**: Uses robot camera when available, falls back to working memory
5. **Unique Filenames**: All saved images have unique timestamps to prevent overwriting

## Testing

A comprehensive test suite is provided in `test_image_injection.py`:

```bash
python test_image_injection.py
```

**Tests Include:**
- Unique filename generation
- Image saving with timestamps
- Filepath uniqueness checking
- Image encoding (numpy arrays and files)
- Message structure validation

## Usage Notes

### For Users

No changes needed! The system works automatically:
- Capture images with the robot camera tool
- Every subsequent LLM call will include that image
- Images are automatically saved to working memory with unique names

### For Developers

To manually save images with unique timestamps:

```python
from utils.image_utils import save_image_with_unique_name

filepath = save_image_with_unique_name(
    image_array=my_image,
    directory="working",
    prefix="custom",
    extension="jpg",
    also_save_as_latest=True
)
```

To access the current image injection behavior:

```python
from tools.robot_tools import get_latest_robot_image

current_image = get_latest_robot_image()  # Returns numpy array or None
```

## File Structure

```
experience-engine/
├── utils/
│   ├── llm_wrapper.py          # NEW: Custom LLM with image injection
│   ├── image_utils.py          # NEW: Image handling utilities
│   └── car.py                  # Existing car control
├── tools/
│   └── robot_tools.py          # UPDATED: Timestamp-based saving
├── agent.py                    # UPDATED: Uses ImageInjectingLLM
├── test_image_injection.py     # NEW: Test suite
└── working/                    # Images saved here
    ├── capture_20251118_143027_456.jpg
    ├── capture_20251118_143028_789.jpg
    └── latest_capture.jpg
```

## Performance Considerations

- **Base64 Encoding**: Images are encoded on each LLM call. For very large images, this may add ~10-50ms overhead
- **Memory**: Latest robot image is kept in memory as a global variable
- **Disk Space**: Each capture creates a new file with timestamp. Working memory should be cleared periodically (happens automatically at session end)

## Future Enhancements

Possible improvements:
- Image compression before encoding to reduce token usage
- Configurable image sources (allow specifying which image to use)
- Image caching to avoid re-encoding the same image multiple times
- Maximum image size/resolution limits
- Automatic cleanup of old timestamped images beyond a certain age

## Compatibility

- **LangChain**: Fully compatible, uses standard message formats
- **OpenAI API**: Uses official multi-modal message format
- **Existing Tools**: No changes needed to existing tools
- **Python Version**: Requires Python 3.9+ (for type hints)

## Troubleshooting

### Images Not Appearing in LLM Calls

1. Check that robot has captured an image recently:
   ```python
   from tools.robot_tools import get_latest_robot_image
   print(get_latest_robot_image() is not None)
   ```

2. Check working memory for images:
   ```bash
   ls -lt working/*.jpg
   ```

3. Enable verbose logging to see image injection:
   ```python
   # Add to llm_wrapper.py in _inject_image_into_messages
   print(f"Injecting image from: {source_description}")
   ```

### "Error encoding image" Messages

- Ensure images are valid and not corrupted
- Check file permissions on working directory
- Verify cv2 is properly installed

### Images Being Overwritten

- This should no longer happen with timestamp-based naming
- If it does, check that `datetime` import is working correctly
- Verify the timestamp format includes milliseconds

## Credits

Implementation based on the sample code in `EXTERNAL_image_tools_SAMPLE.py` and adapted for automatic injection into all LLM messages.

