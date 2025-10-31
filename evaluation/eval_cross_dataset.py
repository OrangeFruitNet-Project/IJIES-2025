# evaluation/eval_cross_dataset.py
import json, sys, os
from ultralytics import YOLO
from pathlib import Path
from utils.reproducibility import set_seed
from evaluation.metrics import mean_sd_ci

def evaluate_model_on_yaml(weights_path, data_yaml):
    model = YOLO(weights_path)
    res = model.val(data=data_yaml, split='test')
    metrics = getattr(res, 'metrics', {}) or {}
    if hasattr(res, 'box'):
        metrics['mAP@0.5'] = getattr(res.box, 'map50', None)
        metrics['mAP@0.5:0.95'] = getattr(res.box, 'map', None)
    return metrics

if __name__ == "__main__":
    set_seed(42)
    conf = json.load(open(sys.argv[1]))
    out = {}

    for c in conf:
        data, w = c['data'], c['weights']
        print(f"Evaluating {w} on {data}")
        metrics = evaluate_model_on_yaml(w, data)
        key = f"{os.path.basename(w)}__on__{Path(data).stem}"
        out[key] = metrics

    # Compute aggregate stats (mean ± SD ± CI)
    values = [m.get('mAP@0.5', 0) for m in out.values() if m.get('mAP@0.5')]
    if values:
        mean_, sd_, ci_ = mean_sd_ci(values)
        out['aggregate'] = {'mean': mean_, 'sd': sd_, 'ci95': ci_}

    json.dump(out, open("eval_summary.json", "w"), indent=2)
    print("✅ Cross-dataset evaluation complete → eval_summary.json")
