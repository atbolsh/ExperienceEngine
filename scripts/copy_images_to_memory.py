import sys, os, shutil

def main():
    if len(sys.argv) < 3:
        print("Usage: copy_images_to_memory.py <dest_folder> <src_image1> [<src_image2> ...]")
        sys.exit(1)
    dest = sys.argv[1]
    srcs = sys.argv[2:]
    os.makedirs(dest, exist_ok=True)
    copied = []
    for src in srcs:
        if not os.path.isfile(src):
            print(f"WARN: source not found: {src}")
            continue
        base = os.path.basename(src)
        dest_path = os.path.join(dest, base)
        try:
            shutil.copyfile(src, dest_path)
            copied.append(dest_path)
        except Exception as e:
            print(f"ERROR copying {src} -> {dest_path}: {e}")
    print("COPIED:" + "\n".join(copied))

if __name__ == "__main__":
    main()
