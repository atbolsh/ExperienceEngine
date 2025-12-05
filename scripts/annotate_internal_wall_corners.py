import cv2
import numpy as np
import os
import glob

WHITE_THRESH = 220  # white field threshold (0-255)
BLACK_THRESH = 50   # wall threshold (0-255)
DOT_RADIUS = 2
MAX_SEARCH = 8  # pixels to step diagonally outward from each corner

WORKING_DIR = os.path.join('working')
LATEST_PATH = os.path.join(WORKING_DIR, 'latest_capture.jpg')
OUTPUT_PATH = os.path.join(WORKING_DIR, 'annotated_view.jpg')


def get_latest_capture():
    # Only read from working/
    if os.path.exists(LATEST_PATH):
        return LATEST_PATH
    imgs = sorted(glob.glob(os.path.join(WORKING_DIR, 'game_capture_*.jpg')))
    return imgs[-1] if imgs else None


def is_on_edge(x, y, w, h, img_w, img_h, tol=0):
    return x <= tol or y <= tol or (x + w) >= img_w - 1 - tol or (y + h) >= img_h - 1 - tol


def load_masks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_mask = cv2.inRange(gray, WHITE_THRESH, 255)
    wall_mask = cv2.inRange(gray, 0, BLACK_THRESH)
    return white_mask, wall_mask


def find_internal_walls(wall_mask):
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = wall_mask.shape
    internal = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if not is_on_edge(x, y, cw, ch, w, h, tol=0):
            internal.append((x, y, cw, ch))
    return internal


def outward_dirs_for_corners():
    return {
        'tl': (-1, -1),
        'tr': ( 1, -1),
        'bl': (-1,  1),
        'br': ( 1,  1),
    }


def corners_from_rect(x, y, w, h):
    return {
        'tl': (x, y),
        'tr': (x + w - 1, y),
        'bl': (x, y + h - 1),
        'br': (x + w - 1, y + h - 1),
    }


def place_purple_dots(img, white_mask, wall_mask, rects):
    h, w, _ = img.shape
    dirs = outward_dirs_for_corners()
    placed = 0
    for (x, y, rw, rh) in rects:
        corners = corners_from_rect(x, y, rw, rh)
        for key, (cx, cy) in corners.items():
            dx, dy = dirs[key]
            chosen = None
            for d in range(1, MAX_SEARCH + 1):
                nx, ny = cx + dx * d, cy + dy * d
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    break
                if white_mask[ny, nx] > 0 and wall_mask[ny, nx] == 0:
                    chosen = (nx, ny)
                    break
            if chosen is not None:
                cv2.circle(img, chosen, DOT_RADIUS, (255, 0, 255), -1)  # purple in BGR
                placed += 1
    return placed


def main():
    src_path = get_latest_capture()
    if src_path is None:
        print('No working/latest_capture.jpg or game_capture_* found.')
        return
    img = cv2.imread(src_path)
    if img is None:
        print(f'Failed to read image: {src_path}')
        return

    white_mask, wall_mask = load_masks(img)
    rects = find_internal_walls(wall_mask)

    annotated = img.copy()
    count = place_purple_dots(annotated, white_mask, wall_mask, rects)

    # Save only inside working/ and only the correctly spelled name
    os.makedirs(WORKING_DIR, exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f'Saved {OUTPUT_PATH} with {count} purple dots. Internal walls: {len(rects)}')

if __name__ == '__main__':
    main()
