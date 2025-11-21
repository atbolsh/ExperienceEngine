#!/usr/bin/env python3
import argparse
import json
import os

# Try OpenCV first, fall back to PIL+NumPy
backend = None
cv2 = None
np = None
Image = None

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    backend = 'opencv'
except Exception:
    try:
        from PIL import Image as PILImage  # type: ignore
        import numpy as np  # type: ignore
        Image = PILImage
        backend = 'pil'
    except Exception:
        backend = None


def detect_red_opencv(img_bgr, min_area_ratio=0.003, morph_kernel=3):
    import cv2
    import numpy as np
    h, w = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Red is at hue near 0 and near 180 in OpenCV (0-179)
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 80])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    if morph_kernel and morph_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area > best_area:
            best_area = area
            best = (x, y, bw, bh)

    area_ratio = best_area / float(w * h) if (w and h) else 0.0
    detected = area_ratio >= min_area_ratio
    return {
        'detected': bool(detected),
        'bbox': list(best) if best else None,
        'area_ratio': area_ratio,
        'image_size': [w, h],
        'mask_count': int(len(contours)),
    }, mask


def detect_red_pil(img_rgb, min_area_ratio=0.003, morph_kernel=None):
    # PIL (0-255 HSV). We will threshold hue close to 0 and 255 for red.
    import numpy as np
    w, h = img_rgb.size
    hsv = img_rgb.convert('HSV')
    hsv_np = np.array(hsv)
    H = hsv_np[:, :, 0]  # 0..255
    S = hsv_np[:, :, 1]
    V = hsv_np[:, :, 2]

    sat_thresh = 80
    val_thresh = 60

    # Red ranges in 0..255 HSV space
    mask1 = ((H <= 15) & (S >= sat_thresh) & (V >= val_thresh))
    mask2 = ((H >= 240) & (S >= sat_thresh) & (V >= val_thresh))
    mask = (mask1 | mask2).astype('uint8') * 255

    # Simple morphology using convolution if available
    try:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area > best_area:
                best_area = area
                best = (x, y, bw, bh)
    except Exception:
        # Fallback: rough bounding box via projection (less accurate)
        ys, xs = np.where(mask > 0)
        if len(xs) and len(ys):
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            best = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
            best_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        else:
            best = None
            best_area = 0

    area_ratio = best_area / float(w * h) if (w and h) else 0.0
    detected = area_ratio >= min_area_ratio
    return {
        'detected': bool(detected),
        'bbox': list(best) if best else None,
        'area_ratio': area_ratio,
        'image_size': [w, h],
    }, mask


def main():
    parser = argparse.ArgumentParser(description='Detect a red bag in an image and optionally save an annotated copy.')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--out', default='', help='Path to save annotated output image (optional)')
    parser.add_argument('--min_area_ratio', type=float, default=0.003, help='Minimum area ratio to consider detection valid')
    args = parser.parse_args()

    if backend is None:
        print(json.dumps({'error': 'No imaging backend available (neither OpenCV nor PIL).'}))
        return

    if backend == 'opencv':
        import cv2
        img = cv2.imread(args.image)
        if img is None:
            print(json.dumps({'error': f'Failed to read image: {args.image}'}))
            return
        result, mask = detect_red_opencv(img, min_area_ratio=args.min_area_ratio)
        if args.out:
            try:
                if result['bbox']:
                    x, y, w, h = result['bbox']
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.imwrite(args.out, img)
                result['annotated_path'] = args.out
            except Exception as e:
                result['annotated_error'] = str(e)
        print(json.dumps(result))
    else:
        # PIL backend
        from PIL import Image as PILImage
        import numpy as np
        try:
            img = PILImage.open(args.image).convert('RGB')
        except Exception as e:
            print(json.dumps({'error': f'Failed to read image: {args.image}', 'detail': str(e)}))
            return
        result, mask = detect_red_pil(img, min_area_ratio=args.min_area_ratio)
        if args.out:
            try:
                # Draw bbox on a copy using PIL
                if result['bbox']:
                    from PIL import ImageDraw
                    ann = img.copy()
                    x, y, w, h = result['bbox']
                    draw = ImageDraw.Draw(ann)
                    draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 0), width=3)
                    ann.save(args.out)
                    result['annotated_path'] = args.out
                else:
                    img.save(args.out)
                    result['annotated_path'] = args.out
            except Exception as e:
                result['annotated_error'] = str(e)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
