import cv2
import numpy as np

def draw_line_to_closest_dot(input_image_path, output_image_path):
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError('Image not found or invalid image format')

    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for green circle and gold dots
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])
    gold_lower = np.array([15, 150, 150])  # Adjust values for gold
    gold_upper = np.array([35, 255, 255])

    # Create masks for green and gold
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    gold_mask = cv2.inRange(hsv, gold_lower, gold_upper)

    # Find contours for green circle
    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('No green circle found')
    green_circle = max(contours, key=cv2.contourArea)  # Assume the largest green object is the circle
    (xg, yg), _ = cv2.minEnclosingCircle(green_circle)

    # Find contours for gold dots
    contours, _ = cv2.findContours(gold_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the closest gold dot
    closest_dot = None
    min_distance = float('inf')
    for contour in contours:
        (x, y), _ = cv2.minEnclosingCircle(contour)
        distance = np.sqrt((xg - x) ** 2 + (yg - y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_dot = (int(x), int(y))

    # Draw a blue line to the closest gold dot
    if closest_dot:
        cv2.line(image, (int(xg), int(yg)), closest_dot, (255, 0, 0), 5)  # Blue line

    # Save the output image
    cv2.imwrite(output_image_path, image)

# Example usage
# draw_line_to_closest_dot('input.jpg', 'output.jpg')