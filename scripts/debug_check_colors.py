import cv2, numpy as np, sys
path = sys.argv[1] if len(sys.argv)>1 else 'working/anotated_view.jpg'
img = cv2.imread(path)
if img is None:
    raise SystemExit('failed to read '+path)
# stats
h,w,_ = img.shape
print('shape', h,w)
# Count near-magenta pixels (bgr)
arr = img.astype(np.int16)
# Tolerances
for tol in [10,20,30,40,60,80,100]:
    diff = np.abs(arr - np.array([255,0,255], dtype=np.int16))
    mask = (diff[:,:,0] <= tol) & (diff[:,:,1] <= tol) & (diff[:,:,2] <= tol)
    cnt = int(mask.sum())
    print('tol', tol, 'magenta_pixels', cnt)
