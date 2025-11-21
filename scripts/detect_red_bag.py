import sys
import json
import cv2
import numpy as np

def detect_red_bag(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        return {"status": "error", "message": f"Could not read image: {image_path}"}

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color range in HSV (two ranges due to hue wrap-around)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = w * h
    min_area = max(400, int(0.003 * img_area))  # at least ~0.3% of image or 400 px

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = min(cw, ch) / max(cw, ch) if max(cw, ch) > 0 else 0
        # Avoid extremely thin shapes (like lines)
        if aspect < 0.15:
            continue
        candidates.append({
            'area': int(area),
            'bbox': [int(x), int(y), int(cw), int(ch)],
            'bbox_area': int(cw * ch)
        })

    found = len(candidates) > 0
    result = {
        "status": "ok",
        "found": found,
        "candidates": sorted(candidates, key=lambda c: c['area'], reverse=True)[:5],
        "image_path": image_path,
        "image_size": [int(w), int(h)]
    }

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Usage: detect_red_bag.py <image_path>"}))
        sys.exit(1)
    image_path = sys.argv[1]
    res = detect_red_bag(image_path)
    print(json.dumps(res))
