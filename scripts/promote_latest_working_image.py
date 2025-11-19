import os
import shutil
import sys

WORKING_DIR = 'working'
EPISODIC_DIR = 'episodic'

# Usage: python promote_latest_working_image.py <memory_name>
# Copies latest image from working/ into episodic/<memory_name>/images/frame_XX.jpg
# Prints the destination filename (e.g., frame_01.jpg)

def ensure_images_dir(memory_name):
    dest_dir = os.path.join(EPISODIC_DIR, memory_name, 'images')
    os.makedirs(dest_dir, exist_ok=True)
    return dest_dir


def find_latest_working_image():
    candidates = []
    for fname in os.listdir(WORKING_DIR):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(WORKING_DIR, fname)
            try:
                mtime = os.path.getmtime(path)
                candidates.append((mtime, path))
            except FileNotFoundError:
                continue
    if not candidates:
        raise FileNotFoundError('No images found in working/')
    candidates.sort(reverse=True)
    return candidates[0][1]


def next_frame_index(dest_dir):
    existing = [f for f in os.listdir(dest_dir) if f.startswith('frame_') and f.lower().endswith('.jpg')]
    max_idx = 0
    for f in existing:
        try:
            num = int(f.split('_')[1].split('.')[0])
            if num > max_idx:
                max_idx = num
        except Exception:
            continue
    return max_idx + 1


def main():
    if len(sys.argv) < 2:
        print('ERROR: memory_name argument required', file=sys.stderr)
        sys.exit(2)
    memory_name = sys.argv[1]

    latest = find_latest_working_image()
    dest_dir = ensure_images_dir(memory_name)
    idx = next_frame_index(dest_dir)
    dest_name = f'frame_{idx:02d}.jpg'
    dest_path = os.path.join(dest_dir, dest_name)

    shutil.copy2(latest, dest_path)

    print(dest_name)

if __name__ == '__main__':
    main()
