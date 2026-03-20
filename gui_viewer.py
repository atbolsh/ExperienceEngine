"""
Non-blocking GUI viewer for environment visualization.

Display modes:
  - terminal: OpenCV window (default), suitable for main.py / CLI.
  - jupyter: No OpenCV thread; images are passed to an optional handler
    (e.g. ipywidgets.Image) registered via set_jupyter_image_handler().

Override with set_display_mode() before initialize_env(), or set DISPLAY_MODE
in select_environment.config (terminal | jupyter).
"""

import os
import cv2
import threading
import numpy as np
from typing import Callable, Optional
import time

# BGR numpy array (same as OpenCV path)
JupyterImageHandler = Callable[[np.ndarray], None]

_display_mode: Optional[str] = None
_jupyter_image_handler: Optional[JupyterImageHandler] = None


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
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Resize image so the smaller side is 400 pixels
            h, w = image.shape[:2]
            min_side = min(h, w)
            
            # Calculate scaling factor to make smaller side 400 pixels
            scale = 400.0 / min_side
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize the image
            self.current_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def stop(self):
        """Stop the viewer thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)


# Global viewer instance
_viewer: Optional[EnvironmentViewer] = None


def load_display_mode_from_config() -> str:
    """Read DISPLAY_MODE from select_environment.config; default terminal."""
    config_path = os.path.join(os.path.dirname(__file__), "select_environment.config")
    if not os.path.exists(config_path):
        return "terminal"
    try:
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DISPLAY_MODE="):
                    value = line.split("=", 1)[1].strip().lower()
                    if value in ("terminal", "jupyter"):
                        return value
        return "terminal"
    except OSError:
        return "terminal"


def get_display_mode() -> str:
    """Active display mode: 'terminal' (OpenCV) or 'jupyter' (widget handler)."""
    global _display_mode
    if _display_mode is None:
        _display_mode = load_display_mode_from_config()
    return _display_mode


def set_display_mode(mode: str) -> None:
    """Force display mode before environment init (e.g. set 'jupyter' in a notebook)."""
    global _display_mode
    m = mode.strip().lower()
    if m not in ("terminal", "jupyter"):
        raise ValueError("display mode must be 'terminal' or 'jupyter'")
    _display_mode = m


def set_jupyter_image_handler(handler: Optional[JupyterImageHandler]) -> None:
    """Register a callback receiving BGR uint8 images when mode is jupyter."""
    global _jupyter_image_handler
    _jupyter_image_handler = handler


def should_open_terminal_window() -> bool:
    """True if OpenCV viewer should be started (terminal mode and enabled)."""
    return get_display_mode() == "terminal"


def visual_updates_enabled(gui_config_true: bool) -> bool:
    """Whether to run viewer init + initial capture (GUI config or jupyter mode)."""
    return gui_config_true or get_display_mode() == "jupyter"


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
        _viewer = None

    if not enabled or get_display_mode() == "jupyter":
        return

    _viewer = EnvironmentViewer(window_name=window_name, enabled=True)


def update_viewer(image: np.ndarray):
    """
    Update the viewer with a new image.
    
    Args:
        image: New image to display (BGR format)
    """
    global _viewer, _jupyter_image_handler

    if get_display_mode() == "jupyter" and _jupyter_image_handler is not None:
        try:
            _jupyter_image_handler(image)
        except Exception:
            pass

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
                if line.startswith("GUI="):
                    value = line.split("=", 1)[1].strip().lower()
                    return value in ["true", "1", "yes", "on"]
        
        return True  # Default to enabled if not found
    except Exception as e:
        print(f"Error reading GUI config: {e}")
        return True  # Default to enabled on error

