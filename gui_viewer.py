"""
Non-blocking GUI viewer for environment visualization.
Displays the current state of the environment (game or car) in a small window.
"""

import cv2
import threading
import numpy as np
from typing import Optional
import time


class EnvironmentViewer:
    """Non-blocking viewer for environment state."""
    
    def __init__(self, window_name: str = "Environment View", enabled: bool = True):
        """
        Initialize the environment viewer.
        
        Args:
            window_name: Name of the OpenCV window
            enabled: Whether the viewer is enabled
        """
        self.window_name = window_name
        self.enabled = enabled
        self.current_image: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        if self.enabled:
            self._start_viewer_thread()
    
    def _start_viewer_thread(self):
        """Start the viewer thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self.thread.start()
    
    def _viewer_loop(self):
        """Main loop for the viewer thread."""
        # Initialize window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 400, 400)
        
        # Create a placeholder image
        placeholder = np.ones((400, 400, 3), dtype=np.uint8) * 50
        cv2.putText(placeholder, "Waiting for image...", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        while self.running:
            with self.lock:
                if self.current_image is not None:
                    display_image = self.current_image.copy()
                else:
                    display_image = placeholder
            
            cv2.imshow(self.window_name, display_image)
            
            # Non-blocking wait (1ms) to update window
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Allow closing with 'q'
                self.stop()
                break
            
            # Small sleep to prevent high CPU usage
            time.sleep(0.01)
        
        # Clean up
        cv2.destroyWindow(self.window_name)
    
    def update_image(self, image: np.ndarray):
        """
        Update the displayed image.
        
        Args:
            image: New image to display (BGR format, as used by OpenCV)
        """
        if not self.enabled:
            return
        
        with self.lock:
            # Ensure image is in the correct format
            if len(image.shape) == 2:
                # Grayscale to BGR
                self.current_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                self.current_image = image
    
    def stop(self):
        """Stop the viewer thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)


# Global viewer instance
_viewer: Optional[EnvironmentViewer] = None


def initialize_viewer(enabled: bool = True, window_name: str = "Environment View"):
    """
    Initialize the global environment viewer.
    
    Args:
        enabled: Whether to enable the viewer
        window_name: Name of the window
    """
    global _viewer
    
    if _viewer is not None:
        _viewer.stop()
    
    _viewer = EnvironmentViewer(window_name=window_name, enabled=enabled)


def update_viewer(image: np.ndarray):
    """
    Update the viewer with a new image.
    
    Args:
        image: New image to display (BGR format)
    """
    global _viewer
    
    if _viewer is not None:
        _viewer.update_image(image)


def close_viewer():
    """Close the viewer."""
    global _viewer
    
    if _viewer is not None:
        _viewer.stop()
        _viewer = None


def is_viewer_enabled() -> bool:
    """Check if the viewer is enabled."""
    global _viewer
    return _viewer is not None and _viewer.enabled


def load_gui_config() -> bool:
    """
    Load the GUI configuration from select_environment.config.
    
    Returns:
        True if GUI should be enabled, False otherwise
    """
    import os
    
    config_path = os.path.join(os.path.dirname(__file__), 'select_environment.config')
    
    if not os.path.exists(config_path):
        return True  # Default to enabled
    
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('GUI='):
                    value = line.split('=', 1)[1].strip().lower()
                    return value in ['true', '1', 'yes', 'on']
        
        return True  # Default to enabled if not found
    except Exception as e:
        print(f"Error reading GUI config: {e}")
        return True  # Default to enabled on error

