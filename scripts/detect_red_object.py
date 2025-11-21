import sys
import json
import cv2
import numpy as np

# Usage: python detect_red_object.py <image_path> [min_area_ratio]
# Outputs JSON: {"found": bool, "area": int, "bbox": [x,y,w,h], "area_ratio": float}

def detect_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red can wrap around the hue edges; use two ranges
    lower_red1 = np.array([0, 90, 90])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 90, 90])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour
    c = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(c))
    x, y, w, h = cv2.boundingRect(c)
    return {"area": area, "bbox": [int(x), int(y), int(w), int(h)], "mask_sum": int(np.sum(mask > 0))}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "image_path required"}))
        sys.exit(1)
    image_path = sys.argv[1]
    min_area_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01  # 1% of image

    img = cv2.imread(image_path)
    if img is None:
        print(json.dumps({"error": f"failed to read image: {image_path}"}))
        sys.exit(1)

    h, w = img.shape[:2]
    total_area = w * h

    det = detect_red(img)
    if det is None:
        print(json.dumps({"found": False, "area": 0, "bbox": None, "area_ratio": 0.0}))
        return

    area_ratio = det["area"] / float(total_area)

    # Optional simple shape heuristic: prefer roughly rectangular, non-skinny
    x, y, bw, bh = det["bbox"]
    aspect = bw / float(bh) if bh > 0 else 0
    shape_ok = 0.3 <= aspect <= 3.0 and det["area"] >= 0.0005 * total_area  # at least 0.05%

    found = area_ratio >= min_area_ratio or (area_ratio >= 0.003 and shape_ok)  # allow smaller if shape ok

    print(json.dumps({
        "found": bool(found),
        "area": int(det["area"]),
        "bbox": det["bbox"],
        "area_ratio": float(area_ratio),
        "aspect": aspect,
        "shape_ok": shape_ok,
        "image_size": [int(w), int(h)]
    }))

if __name__ == "__main__":
    main()
