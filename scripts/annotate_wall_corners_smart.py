import cv2
import numpy as np
import os
import glob

# I/O strictly limited to working/
WORKING_DIR = 'working'
INPUT_PRIMARY = os.path.join(WORKING_DIR, 'latest_capture.jpg')
OUTPUT_CANON = os.path.join(WORKING_DIR, 'annotated_view.jpg')

# Tunable params
WHITE_THRESH = 220
BLACK_THRESH = 50
MIN_WALL_AREA = 8
EDGE_MARGIN = 2
DOT_RADIUS = 2
PURPLE_BGR = (255, 0, 255)
MAX_SEARCH = 6


def get_input_path():
    if os.path.exists(INPUT_PRIMARY):
        return INPUT_PRIMARY
    imgs = sorted(glob.glob(os.path.join(WORKING_DIR, 'game_capture_*.jpg')))
    return imgs[-1] if imgs else None


def first_white_point(px, py, dx, dy, white_mask, wall_mask, w, h):
    for s in range(1, MAX_SEARCH + 1):
        nx = px + dx * s
        ny = py + dy * s
        if nx < 0 or ny < 0 or nx >= w or ny >= h:
            return None
        if white_mask[ny, nx] and wall_mask[ny, nx] == 0:
            return (nx, ny)
    return None


def main():
    os.makedirs(WORKING_DIR, exist_ok=True)
    src = get_input_path()
    if src is None:
        raise SystemExit('No working/latest_capture.jpg or game_capture_* found')

    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit('Failed to read ' + src)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    white_mask = (gray > WHITE_THRESH)
    wall_mask = (gray < BLACK_THRESH).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def is_internal_rect(x, y, bw, bh):
        if x <= EDGE_MARGIN or y <= EDGE_MARGIN:
            return False
        if x + bw >= w - EDGE_MARGIN or y + bh >= h - EDGE_MARGIN:
            return False
        return True

    placed = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_WALL_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if not is_internal_rect(x, y, bw, bh):
            continue

        corners_dirs = [
            ((x, y), (-1, -1)),
            ((x + bw - 1, y), (1, -1)),
            ((x, y + bh - 1), (-1, 1)),
            ((x + bw - 1, y + bh - 1), (1, 1)),
        ]

        for (cx, cy), (dx, dy) in corners_dirs:
            pt = first_white_point(cx, cy, dx, dy, white_mask, wall_mask, w, h)
            if pt is not None:
                cv2.circle(img, pt, DOT_RADIUS, PURPLE_BGR, thickness=-1, lineType=cv2.LINE_AA)
                placed += 1

    cv2.imwrite(OUTPUT_CANON, img)
    print(f'Saved {OUTPUT_CANON} with {placed} purple dots')

if __name__ == '__main__':
    main()
