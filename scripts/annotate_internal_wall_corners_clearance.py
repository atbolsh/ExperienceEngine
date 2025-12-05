import cv2
import numpy as np
import os
import glob

# Thresholds and knobs
WHITE_THRESH = 220   # white field threshold (0-255)
BLACK_THRESH = 50    # wall threshold (0-255)
DOT_RADIUS   = 2
RADIUS_LIMIT_MULT = 1.2  # hard stop: do NOT search beyond 1.2 * estimated agent radius
DEFAULT_AGENT_RADIUS = 5
AGENT_RADIUS_CLAMP = (3, 9)  # min, max radius to keep estimates reasonable for 64x64 game

WORKING_DIR = os.path.join('working')
LATEST_PATH = os.path.join(WORKING_DIR, 'latest_capture.jpg')
OUTPUT_PATH = os.path.join(WORKING_DIR, 'annotated_view.jpg')  # single canonical output


def get_latest_capture():
    # Only ever read from working/
    if os.path.exists(LATEST_PATH):
        return LATEST_PATH
    imgs = sorted(glob.glob(os.path.join(WORKING_DIR, 'game_capture_*.jpg')))
    return imgs[-1] if imgs else None


def load_masks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_mask = cv2.inRange(gray, WHITE_THRESH, 255)
    wall_mask  = cv2.inRange(gray, 0, BLACK_THRESH)
    return white_mask, wall_mask


def is_on_edge(x, y, w, h, img_w, img_h, tol=0):
    return x <= tol or y <= tol or (x + w) >= img_w - 1 - tol or (y + h) >= img_h - 1 - tol


def find_internal_walls(wall_mask):
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = wall_mask.shape
    rects = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if not is_on_edge(x, y, cw, ch, w, h, tol=0):
            rects.append((x, y, cw, ch))
    return rects


def corners_from_rect(x, y, w, h):
    return {
        'tl': (x, y),
        'tr': (x + w - 1, y),
        'bl': (x, y + h - 1),
        'br': (x + w - 1, y + h - 1),
    }


def corner_quadrant_signs():
    # Required signs for offset (dx,dy) relative to the corner to ensure we move away from both edges
    return {
        'tl': (-1, -1),
        'tr': ( 1, -1),
        'bl': (-1,  1),
        'br': ( 1,  1),
    }


def estimate_agent_radius(img):
    # HSV thresholding for the green agent
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 80, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return DEFAULT_AGENT_RADIUS
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area <= 0:
        return DEFAULT_AGENT_RADIUS
    r = (area / np.pi) ** 0.5
    r = int(round(r))
    # Clamp
    r = max(AGENT_RADIUS_CLAMP[0], min(AGENT_RADIUS_CLAMP[1], r))
    return r


def make_collision_mask(wall_mask, agent_radius):
    # Dilate walls by agent radius so any center inside this mask would collide
    ksize = int(max(1, agent_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ksize+1, 2*ksize+1))
    dil = cv2.dilate(wall_mask, kernel, iterations=1)
    return dil


def search_point_in_quadrant(cx, cy, signx, signy, d_start, d_limit, white_mask, wall_mask, collision_mask):
    h, w = white_mask.shape
    # Search ring-by-ring from d_start up to and including d_limit
    for d in range(d_start, d_limit + 1):
        x_min, x_max = cx - d, cx + d
        y_min, y_max = cy - d, cy + d
        candidates = []
        # top and bottom edges
        for x in range(x_min, x_max + 1):
            candidates.append((x, y_min))
            candidates.append((x, y_max))
        # left and right edges
        for y in range(y_min + 1, y_max):
            candidates.append((x_min, y))
            candidates.append((x_max, y))

        for nx, ny in candidates:
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            dx = nx - cx
            dy = ny - cy
            # Must be in the requested quadrant (away from both edges)
            if np.sign(dx) != np.sign(signx) or np.sign(dy) != np.sign(signy):
                continue
            if white_mask[ny, nx] > 0 and wall_mask[ny, nx] == 0 and collision_mask[ny, nx] == 0:
                return (nx, ny)
    return None


def place_purple_dots(img, white_mask, wall_mask, rects, agent_radius):
    placed = 0
    collision_mask = make_collision_mask(wall_mask, agent_radius)

    # Minimum safe offset so the agent centered on the dot would not touch walls
    d_start = max(2, int(agent_radius) + 1)
    # Hard stop distance per requirement: do not search beyond 1.2 * estimated agent radius
    d_limit = int(np.floor(RADIUS_LIMIT_MULT * float(agent_radius)))

    signs = corner_quadrant_signs()
    for (x, y, rw, rh) in rects:
        corners = corners_from_rect(x, y, rw, rh)
        for key, (cx, cy) in corners.items():
            signx, signy = signs[key]
            if d_start > d_limit:
                # Not enough room to place a safe dot for this corner
                continue
            pt = search_point_in_quadrant(
                cx, cy, signx, signy, d_start, d_limit,
                white_mask, wall_mask, collision_mask
            )
            if pt is not None:
                cv2.circle(img, pt, DOT_RADIUS, (255, 0, 255), -1)  # purple (BGR)
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
    agent_radius = estimate_agent_radius(img)

    annotated = img.copy()
    count = place_purple_dots(annotated, white_mask, wall_mask, rects, agent_radius)

    # Always and only save into working/ with the required canonical filename
    os.makedirs(WORKING_DIR, exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, annotated)

    print(f"Saved {OUTPUT_PATH} with {count} purple dots. Internal walls: {len(rects)}. Agent radius: {agent_radius}")


if __name__ == '__main__':
    main()
