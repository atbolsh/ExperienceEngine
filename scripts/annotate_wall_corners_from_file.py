import cv2
import numpy as np
import os
import glob

WORKING_DIR = 'working'
INPUT_PATH_PRIMARY = os.path.join(WORKING_DIR, 'latest_capture.jpg')
OUTPUT_CANON = os.path.join(WORKING_DIR, 'annotated_view.jpg')

# Parameters
WHITE_THRESH = 245
BLACK_THRESH = 30
MIN_WALL_AREA = 10
EDGE_MARGIN = 1
DOT_RADIUS = 2
OFFSET = DOT_RADIUS + 1
PURPLE_BGR = (255, 0, 255)

def get_input_path():
    if os.path.exists(INPUT_PATH_PRIMARY):
        return INPUT_PATH_PRIMARY
    imgs = sorted(glob.glob(os.path.join(WORKING_DIR, 'game_capture_*.jpg')))
    return imgs[-1] if imgs else None


def is_white_and_not_wall(px, py, white_mask, wall_mask, w, h):
    if px < 0 or py < 0 or px >= w or py >= h:
        return False
    return white_mask[py, px] == 255 and wall_mask[py, px] == 0


def main():
    os.makedirs(WORKING_DIR, exist_ok=True)
    src = get_input_path()
    if src is None:
        raise SystemExit('No working/latest_capture.jpg or game_capture_* found')

    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit('Failed to read input image: '+src)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    white_mask = (gray > WHITE_THRESH).astype(np.uint8) * 255
    wall_mask = (gray < BLACK_THRESH).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def is_internal(x, y, bw, bh):
        if x <= EDGE_MARGIN or y <= EDGE_MARGIN:
            return False
        if x + bw >= w - EDGE_MARGIN or y + bh >= h - EDGE_MARGIN:
            return False
        return True

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
            if is_white_and_not_wall(px, py, white_mask, wall_mask, w, h):
                cv2.circle(img, (px, py), DOT_RADIUS, PURPLE_BGR, thickness=-1, lineType=cv2.LINE_AA)

    cv2.imwrite(OUTPUT_CANON, img)
    print(f'Saved {OUTPUT_CANON}')

if __name__ == '__main__':
    main()
