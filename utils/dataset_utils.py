# utils/dataset_utils.py
import csv, json, os, shutil, random
from pathlib import Path
from sklearn.model_selection import train_test_split

def read_manifest(manifest_csv):
    with open(manifest_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def split_dataset(manifest_rows, out_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    imgs = manifest_rows.copy()
    random.shuffle(imgs)
    n = len(imgs)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    train = imgs[:t]
    val = imgs[t:t+v]
    test = imgs[t+v:]
    # create folders and copy images if local files exist; otherwise only create split json
    splits = {'train': train, 'val': val, 'test': test}
    for k, arr in splits.items():
        Path(out_dir, k).mkdir(parents=True, exist_ok=True)
        # assume images exist at 'source_path' key or else manifest has image_id mapping
        # Caller may implement copying externally if images remote.
    return splits

# Very simple YOLO txt writer (single-class)
def write_yolo_annotations(split_rows, images_root, out_images_dir, out_labels_dir):
    """
    Expects each row to have 'image_id' and bounding box columns; adjust to your manifest.
    """
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)
    Path(out_labels_dir).mkdir(parents=True, exist_ok=True)
    for r in split_rows:
        src = Path(images_root) / r['image_id']
        dst = Path(out_images_dir) / r['image_id']
        if src.exists():
            shutil.copy(src, dst)
        # assume r contains bbox list in JSON under 'bboxes', each bbox: [x,y,w,h] px
        bboxes = json.loads(r.get('bboxes', '[]'))
        h, w = r.get('height'), r.get('width')
        # write YOLO label file
        label_path = Path(out_labels_dir) / (Path(r['image_id']).stem + '.txt')
        with open(label_path, 'w') as f:
            for bb in bboxes:
                x,y,bw,bh = bb
                # normalize
                xc = (x + bw/2) / w
                yc = (y + bh/2) / h
                nx = bw / w
                ny = bh / h
                # single class 0
                f.write(f"0 {xc:.6f} {yc:.6f} {nx:.6f} {ny:.6f}\n")
