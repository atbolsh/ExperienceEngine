import re
import base64
import sys
import os
from io import BytesIO

import numpy as np

# Try OpenCV, fall back to PIL for I/O/drawing if OpenCV isn't available
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except Exception:
    HAS_PIL = False

DATA_URL = """
DATA_IMAGE_PLACEHOLDER
""".strip()

WORKING_DIR = 'working'
OUTPUT_CANON = os.path.join(WORKING_DIR, 'annotated_view.jpg')

# Small helper: decode data URL or raw base64 to bytes

def decode_data_url(data_url: str) -> bytes:
    m = re.search(r'base64,([A-Za-z0-9+/=\n\r]+)', data_url)
    if m:
        b64 = m.group(1)
    else:
        # Assume it's pure base64 already
        b64 = data_url
    return base64.b64decode(b64)

# Simple color helpers
PURPLE_BGR = (255, 0, 255)
PURPLE_RGB = (255, 0, 255)

# Detection thresholds (tuned conservatively)
WHITE_THRESH = 245    # grayscale > WHITE_THRESH => white playing field
BLACK_THRESH = 30     # grayscale < BLACK_THRESH => wall
MIN_WALL_AREA = 10    # min contour area to consider a wall
EDGE_MARGIN = 1       # exclude contours touching image border
DOT_RADIUS = 2        # purple ball radius
OFFSET = DOT_RADIUS + 1  # just barely inside white field


def draw_with_cv2(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Masks
    white_mask = (gray > WHITE_THRESH).astype(np.uint8) * 255
    wall_mask = (gray < BLACK_THRESH).astype(np.uint8) * 255

    # Clean masks a bit
    kernel = np.ones((3, 3), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find separate wall components
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def is_internal(x, y, bw, bh):
        # Exclude any wall touching outer image boundary (likely outer border)
        if x <= EDGE_MARGIN or y <= EDGE_MARGIN:
            return False
        if x + bw >= w - EDGE_MARGIN or y + bh >= h - EDGE_MARGIN:
            return False
        return True

    def is_white_and_not_wall(cx, cy):
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return False
        return white_mask[cy, cx] == 255 and wall_mask[cy, cx] == 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_WALL_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if not is_internal(x, y, bw, bh):
            continue

        corners = [
            (x, y, -OFFSET, -OFFSET),
            (x + bw - 1, y, +OFFSET, -OFFSET),
            (x, y + bh - 1, -OFFSET, +OFFSET),
            (x + bw - 1, y + bh - 1, +OFFSET, +OFFSET),
        ]

        for (cx, cy, dx, dy) in corners:
            px = int(round(cx + dx))
            py = int(round(cy + dy))
            if is_white_and_not_wall(px, py):
                cv2.circle(img_bgr, (px, py), DOT_RADIUS, PURPLE_BGR, thickness=-1, lineType=cv2.LINE_AA)

    return img_bgr


def draw_with_pil(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    gray = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]).astype(np.uint8)
    white_mask = (gray > WHITE_THRESH)
    wall_mask = (gray < BLACK_THRESH)

    # Attempt connected components using scipy if present
    components = []
    try:
        from scipy.ndimage import label, find_objects
        labeled, n = label(wall_mask.astype(np.uint8))
        slices = find_objects(labeled)
        for sl in slices:
            if sl is None:
                continue
            ys, xs = sl
            y0, y1 = ys.start, ys.stop
            x0, x1 = xs.start, xs.stop
            area = (y1 - y0) * (x1 - x0)
            if area >= MIN_WALL_AREA:
                components.append((x0, y0, x1 - x0, y1 - y0))
    except Exception:
        ys, xs = np.where(wall_mask)
        if ys.size > 0:
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            components.append((x0, y0, x1 - x0, y1 - y0))

    from PIL import Image, ImageDraw
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    def is_internal(x, y, bw, bh):
        if x <= EDGE_MARGIN or y <= EDGE_MARGIN:
            return False
        if x + bw >= w - EDGE_MARGIN or y + bh >= h - EDGE_MARGIN:
            return False
        return True

    def is_white_and_not_wall(cx, cy):
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return False
        return white_mask[cy, cx] and (not wall_mask[cy, cx])

    for (x, y, bw, bh) in components:
        if not is_internal(x, y, bw, bh):
            continue
        corners = [
            (x, y, -OFFSET, -OFFSET),
            (x + bw - 1, y, +OFFSET, -OFFSET),
            (x, y + bh - 1, -OFFSET, +OFFSET),
            (x + bw - 1, y + bh - 1, +OFFSET, +OFFSET),
        ]
        for (cx, cy, dx, dy) in corners:
            px = int(round(cx + dx))
            py = int(round(cy + dy))
            if is_white_and_not_wall(px, py):
                r = DOT_RADIUS
                draw.ellipse([px - r, py - r, px + r, py + r], fill=PURPLE_RGB)

    return np.array(img_pil)


def main():
    os.makedirs(WORKING_DIR, exist_ok=True)

    try:
        data = decode_data_url(DATA_URL)
    except Exception as e:
        print(f"Failed to decode input image: {e}")
        sys.exit(1)

    img = None
    if HAS_CV2:
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_cv is not None:
                img = img_cv
        except Exception:
            img = None

    if img is None:
        if not HAS_PIL:
            print("Neither cv2 nor PIL available to load image.")
            sys.exit(1)
        try:
            from PIL import Image
            pil = Image.open(BytesIO(data)).convert('RGB')
            img = np.array(pil)[:, :, ::-1] if HAS_CV2 else np.array(pil)  # to BGR if cv2 present later
        except Exception as e:
            print(f"Failed to load image with PIL: {e}")
            sys.exit(1)

    if HAS_CV2:
        annotated = draw_with_cv2(img.copy())
        cv2.imwrite(OUTPUT_CANON, annotated)
    else:
        if not HAS_PIL:
            print("No drawing backend available (cv2/PIL missing).")
            sys.exit(1)
        if img.shape[2] == 3:
            img_rgb = img[:, :, ::-1]
        else:
            img_rgb = img
        annotated = draw_with_pil(img_rgb)
        from PIL import Image
        Image.fromarray(annotated).save(OUTPUT_CANON)

    print(f"Saved {OUTPUT_CANON}")

if __name__ == '__main__':
    main()
