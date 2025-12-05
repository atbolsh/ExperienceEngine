import cv2
import numpy as np
import sys

# Default to working/annotated image paths
PATH = sys.argv[1] if len(sys.argv) > 1 else 'working/anotated_view.jpg'
img = cv2.imread(PATH)
if img is None:
    raise SystemExit('failed to read ' + PATH)
# BGR target purple
target = np.array([255,0,255], dtype=np.int16)
arr = img.astype(np.int16)
diff = np.abs(arr - target)
mask = (diff[:,:,0] <= 10) & (diff[:,:,1] <= 10) & (diff[:,:,2] <= 10)
mask_u8 = mask.astype(np.uint8)*255
# Morph open to clean
kernel = np.ones((3,3), np.uint8)
mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
# Count connected components
num_labels, labels = cv2.connectedComponents(mask_u8)
# num_labels includes background as 0
print('purple_dot_count', max(0, num_labels-1))
