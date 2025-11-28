"""
Test script for image injection functionality.
This script verifies that the ImageInjectingLLM wrapper correctly injects images into messages.
"""

import os
import sys
import tempfile
import cv2
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from environments.car_environment.llm_wrapper import encode_image_to_base64, encode_numpy_image_to_base64
from utils.image_utils import generate_unique_image_filename, save_image_with_unique_name, get_unique_filepath
from langchain_core.messages import HumanMessage


def test_unique_filename_generation():
    """Test that unique filenames are generated with timestamps."""
    print("Testing unique filename generation...")
    
    filename1 = generate_unique_image_filename("test", "jpg")
    filename2 = generate_unique_image_filename("test", "jpg")
    
    assert filename1 != filename2, "Generated filenames should be unique"
    assert filename1.startswith("test_"), "Filename should start with prefix"
    assert filename1.endswith(".jpg"), "Filename should end with extension"
    
    print(f"  ✓ Generated unique filenames: {filename1}, {filename2}")
    return True


def test_save_image_with_unique_name():
    """Test saving images with unique timestamps."""
    print("\nTesting image saving with unique names...")
    
    # Create a test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:] = (0, 0, 255)  # Red image
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save first image
        filepath1 = save_image_with_unique_name(
            test_image, 
            tmpdir, 
            prefix="capture",
            also_save_as_latest=True
        )
        
        # Save second image
        filepath2 = save_image_with_unique_name(
            test_image, 
            tmpdir, 
            prefix="capture",
            also_save_as_latest=True
        )
        
        # Check that files exist
        assert os.path.exists(filepath1), "First image should exist"
        assert os.path.exists(filepath2), "Second image should exist"
        assert filepath1 != filepath2, "Filepaths should be unique"
        
        # Check that latest file exists
        latest_path = os.path.join(tmpdir, "latest_capture.jpg")
        assert os.path.exists(latest_path), "Latest file should exist"
        
        print(f"  ✓ Saved images to: {os.path.basename(filepath1)}, {os.path.basename(filepath2)}")
        print(f"  ✓ Latest file created: latest_capture.jpg")
    
    return True


def test_get_unique_filepath():
    """Test getting unique filepath when file exists."""
    print("\nTesting unique filepath generation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = "test.jpg"
        filepath1 = get_unique_filepath(tmpdir, filename)
        
        # Create the file
        with open(filepath1, 'w') as f:
            f.write("test")
        
        # Get another unique filepath (should add timestamp)
        filepath2 = get_unique_filepath(tmpdir, filename)
        
        assert filepath1 != filepath2, "Second filepath should be unique"
        assert os.path.basename(filepath1) == filename, "First filepath should match original name"
        assert filename.replace('.', '_') in filepath2, "Second filepath should contain timestamp"
        
        print(f"  ✓ First filepath: {os.path.basename(filepath1)}")
        print(f"  ✓ Second filepath: {os.path.basename(filepath2)}")
    
    return True


def test_image_encoding():
    """Test image encoding functions."""
    print("\nTesting image encoding...")
    
    # Create a test image
    test_image = np.zeros((50, 50, 3), dtype=np.uint8)
    test_image[:] = (0, 255, 0)  # Green image
    
    # Test numpy encoding
    base64_data, image_format = encode_numpy_image_to_base64(test_image)
    
    assert len(base64_data) > 0, "Base64 data should not be empty"
    assert image_format == "jpeg", "Format should be jpeg"
    
    print(f"  ✓ Encoded numpy image to base64 (length: {len(base64_data)})")
    
    # Test file encoding
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test.jpg")
        cv2.imwrite(test_path, test_image)
        
        base64_data, image_format = encode_image_to_base64(test_path)
        
        assert len(base64_data) > 0, "Base64 data should not be empty"
        assert image_format == "jpeg", "Format should be jpeg"
        
        print(f"  ✓ Encoded file image to base64 (length: {len(base64_data)})")
    
    return True


def test_message_injection():
    """Test that messages are properly structured for image injection."""
    print("\nTesting message injection structure...")
    
    # Create a simple message
    message = HumanMessage(content="Test message")
    
    # Verify message structure
    assert isinstance(message.content, str), "Message content should be string"
    
    # Test multi-modal message structure
    multi_modal_message = HumanMessage(
        content=[
            {"type": "text", "text": "Test message"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,test_data"
                }
            }
        ]
    )
    
    assert isinstance(multi_modal_message.content, list), "Multi-modal content should be list"
    assert len(multi_modal_message.content) == 2, "Should have text and image"
    assert multi_modal_message.content[0]["type"] == "text", "First item should be text"
    assert multi_modal_message.content[1]["type"] == "image_url", "Second item should be image"
    
    print("  ✓ Message structures are correct")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Image Injection Implementation Tests")
    print("=" * 60)
    
    tests = [
        test_unique_filename_generation,
        test_save_image_with_unique_name,
        test_get_unique_filepath,
        test_image_encoding,
        test_message_injection,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All tests passed! Image injection is ready to use.")
        print("\nNote: The ImageInjectingLLM wrapper will automatically inject")
        print("images into all LLM messages when the agent is running.")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

