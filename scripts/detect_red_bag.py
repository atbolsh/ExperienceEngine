import sys, argparse, json, os
import cv2
import numpy as np


def compute_confidence(mask, bbox, contour, image_shape):
    x, y, w, h = bbox
    if w == 0 or h == 0:
        return 0.0
    roi = mask[y:y+h, x:x+w]
    red_pixels = int(np.count_nonzero(roi))
    box_area = w * h
    red_density = red_pixels / max(1, box_area)

    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    cnt_area = cv2.contourArea(contour)
    solidity = cnt_area / max(1.0, hull_area)

    # Extent
    extent = cnt_area / max(1, box_area)

    # Relative size
    img_h, img_w = image_shape[:2]
    img_area = img_w * img_h
    size_ratio = cnt_area / max(1, img_area)

    # Combine heuristics into a confidence score
    # Weights tuned heuristically
    conf = (
        0.45 * min(1.0, red_density) +
        0.25 * min(1.0, solidity) +
        0.20 * min(1.0, extent / 0.8) +
        0.10 * min(1.0, size_ratio / 0.02)
    )
    return float(max(0.0, min(1.0, conf)))


def detect_red_bag(image_path, out_path=None, debug=False):
    if not os.path.exists(image_path):
        return {"detected": False, "error": f"Image not found: {image_path}"}

    img = cv2.imread(image_path)
    if img is None:
        return {"detected": False, "error": f"Failed to read image: {image_path}"}

    img_h, img_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red in HSV can wrap: use two ranges
    # Tune thresholds to be moderately inclusive
    lower1 = np.array([0, 80, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 80])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphology to clean noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_h * img_w
    min_area = max(500, int(0.0015 * img_area))  # ignore tiny specks

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(1.0, h)
        if not (0.35 <= aspect <= 3.0):
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(1.0, hull_area)
        if solidity < 0.55:
            continue
        extent = area / max(1, w*h)
        if extent < 0.25:
            continue
        candidates.append((area, (x, y, w, h), cnt))

    if not candidates:
        result = {
            "detected": False,
            "reason": "No sizable red bag-like region found",
            "bbox": None,
            "area_ratio": 0.0,
            "confidence": 0.0
        }
        return result

    # Pick largest candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    area, bbox, cnt = candidates[0]

    area_ratio = float(area / img_area)
    confidence = compute_confidence(mask, bbox, cnt, img.shape)

    result = {
        "detected": True,
        "bbox": {
            "x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]), "h": int(bbox[3])
        },
        "area_ratio": area_ratio,
        "confidence": confidence
    }

    if out_path is not None:
        vis = img.copy()
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.drawContours(vis, [cnt], -1, (255, 0, 0), 2)
        label = f"Red bag conf={confidence:.2f} area={area_ratio*100:.1f}%"
        cv2.putText(vis, label, (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, vis)
        result["annotated_image"] = out_path

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", required=False, default=None, help="Path to save annotated image")
    args = parser.parse_args()

    res = detect_red_bag(args.image, args.out)
    print(json.dumps(res))
