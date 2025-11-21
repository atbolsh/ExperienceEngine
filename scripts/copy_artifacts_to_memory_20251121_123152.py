#!/usr/bin/env python3
import os, shutil, json

def main():
    dst_dir = 'episodic/red_bag_detection_custom_script_20251121_123152'
    os.makedirs(dst_dir, exist_ok=True)
    src_raw = 'working/latest_capture.jpg'
    src_ann = 'working/latest_capture_annotated.jpg'
    dst_raw = os.path.join(dst_dir, 'scene.jpg')
    dst_ann = os.path.join(dst_dir, 'scene_annotated.jpg')

    out = {'copied': [], 'missing': []}
    for s, d in [(src_raw, dst_raw), (src_ann, dst_ann)]:
        if os.path.exists(s):
            shutil.copy2(s, d)
            out['copied'].append({'src': s, 'dst': d})
        else:
            out['missing'].append(s)

    print(json.dumps(out))

if __name__ == '__main__':
    main()
