import cv2, numpy as np, sys
path = sys.argv[1] if len(sys.argv)>1 else 'working/annotated_view.jpg'
img = cv2.imread(path)
if img is None:
    raise SystemExit('failed to read '+path)
# Count magenta-like pixels and cluster
bgr = img.astype(np.int16)
target = np.array([255,0,255], dtype=np.int16)
diff = np.abs(bgr - target)
mask = (diff[:,:,0] <= 10) & (diff[:,:,1] <= 10) & (diff[:,:,2] <= 10)
mask_u8 = mask.astype(np.uint8)*255
kernel = np.ones((3,3), np.uint8)
mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
num_labels, labels = cv2.connectedComponents(mask_u8)
print('purple_dot_count', max(0, num_labels-1))
