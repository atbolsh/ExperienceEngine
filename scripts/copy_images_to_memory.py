import sys, os, shutil

def main():
    if len(sys.argv) < 3:
        print('Usage: copy_images_to_memory.py <dest_dir> <src_file1> [<src_file2> ...]')
        sys.exit(1)
    dest_dir = sys.argv[1]
    src_files = sys.argv[2:]
    os.makedirs(dest_dir, exist_ok=True)
    results = []
    for src in src_files:
        if not os.path.isfile(src):
            results.append(f'MISSING:{src}')
            continue
        base = os.path.basename(src)
        dest_path = os.path.join(dest_dir, base)
        shutil.copy2(src, dest_path)
        results.append(f'COPIED:{src}->${dest_path}')
    print('\n'.join(results))

if __name__ == '__main__':
    main()
