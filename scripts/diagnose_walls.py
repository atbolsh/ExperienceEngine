import cv2, numpy as np, os, glob

WORKING_DIR = 'working'
INPUT_PRIMARY = os.path.join(WORKING_DIR, 'latest_capture.jpg')

# Choose input strictly from working/
def get_input_path():
    if os.path.exists(INPUT_PRIMARY):
        return INPUT_PRIMARY
    imgs = sorted(glob.glob(os.path.join(WORKING_DIR, 'game_capture_*.jpg')))
    return imgs[-1] if imgs else None

src = get_input_path()
if src is None:
    raise SystemExit('No working/latest_capture.jpg or game_capture_* found')

img = cv2.imread(src, cv2.IMREAD_COLOR)
if img is None:
    raise SystemExit('failed to read '+src)

h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# try a range of thresholds
for BLACK_THRESH in [20, 30, 40, 60, 80, 100, 120]:
    white_mask = (gray > 240).astype(np.uint8)*255
    wall_mask = (gray < BLACK_THRESH).astype(np.uint8)*255
    kernel = np.ones((3,3), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    internals = 0
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue
        x,y,bw,bh = cv2.boundingRect(cnt)
        if x<=1 or y<=1 or x+bw>=w-1 or y+bh>=h-1:
            continue
        internals += 1
        boxes.append((x,y,bw,bh))
    print('BLACK_THRESH', BLACK_THRESH, 'contours', len(contours), 'internal', internals, 'boxes', boxes[:5])

# save masks for visual check strictly into working/
os.makedirs(WORKING_DIR, exist_ok=True)
cv2.imwrite(os.path.join(WORKING_DIR, 'diagnose_gray.jpg'), gray)
cv2.imwrite(os.path.join(WORKING_DIR, 'diagnose_white_mask.jpg'), (gray>240).astype(np.uint8)*255)
cv2.imwrite(os.path.join(WORKING_DIR, 'diagnose_wall_mask_80.jpg'), (gray<80).astype(np.uint8)*255)
print('Saved diagnostic images into working/')
