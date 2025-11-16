import cv2
import numpy as np

# Function to detect the green circle and gold dots and draw a line

def draw_line_on_image(input_image_path, output_image_path):
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError('Image not found or unable to load.')

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for green circle
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])

    # Define color range for gold dots
    gold_lower = np.array([20, 100, 100])
    gold_upper = np.array([30, 255, 255])

    # Create masks for green circle and gold dots
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_gold = cv2.inRange(hsv, gold_lower, gold_upper)

    # Find contours for green circle
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_green:
        raise ValueError('No green circle found.')
    green_circle = max(contours_green, key=cv2.contourArea)
    ((x_green, y_green), radius) = cv2.minEnclosingCircle(green_circle)

    # Find contours for gold dots
    contours_gold, _ = cv2.findContours(mask_gold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_gold:
        raise ValueError('No gold dots found.')

    # Draw a line from the center of the green circle to each gold dot
    for gold_dot in contours_gold:
        ((x_gold, y_gold), radius) = cv2.minEnclosingCircle(gold_dot)
        # Draw a thick blue line
        cv2.line(image, (int(x_green), int(y_green)), (int(x_gold), int(y_gold)), (255, 0, 0), thickness=5)

    # Save the result
    cv2.imwrite(output_image_path, image)

    return output_image_path

# Example usage
# draw_line_on_image('input.jpg', 'output.jpg')