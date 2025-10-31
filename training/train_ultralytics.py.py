# training/train_ultralytics.py
import os, argparse, yaml, json, math
from ultralytics import YOLO
from pathlib import Path
from utils.reproducibility import set_seed


def train_model(cfg):
    model_name = cfg['model_name']  # e.g., 'yolov8n.pt' or 'yolov5s.pt'
    data_yaml = cfg['data_yaml']    # path to dataset yaml file
    epochs = cfg.get('epochs', 200)
    seed = cfg.get('seed', 42)
    exp_name = cfg.get('exp_name', 'exp')

    set_seed(seed)

    model = YOLO(model_name)  # load pretrained YOLO model

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=cfg.get('imgsz', 640),
        batch=cfg.get('batch', 16),
        project=cfg.get('project', 'runs/train'),
        name=exp_name,
        optimizer=cfg.get('optimizer', 'Adam'),
        lr0=cfg.get('lr', 0.001),
        workers=cfg.get('workers', 4),
        seed=seed,
        patience=cfg.get('patience', 25),
        save_period=cfg.get('save_period', 10)
    )

    out = {
        'weights': str(Path(cfg.get('project', 'runs/train')) / exp_name / 'weights' / 'best.pt'),
        'metrics': getattr(results, 'metrics', None)
    }
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='YAML config path with model/data/epochs etc.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # ✅ Load YAML config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    runs = []
    for seed in cfg.get('seeds', [42, 84, 126]):
        cfg['seed'] = seed
        cfg['exp_name'] = f"{cfg.get('base_exp_name', 'exp')}_s{seed}"
        print(f"🚀 Training: {cfg['exp_name']}")
        res = train_model(cfg)
        runs.append(res)

    # ✅ Save summary as JSON (for evaluation compatibility)
    project_dir = Path(cfg.get('project', 'runs/train'))
    project_dir.mkdir(parents=True, exist_ok=True)

    summary_path = project_dir / f"{cfg.get('base_exp_name', 'exp')}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(runs, f, indent=2)

    print(f"✅ Training summary saved to: {summary_path}")
