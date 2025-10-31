# utils/logger.py
"""
Unified experiment logging utility for OrangeFruitNet experiments.
Supports both Weights & Biases (wandb) and local logging fallback.
"""

import os
import json
import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARN] wandb not installed. Logging locally instead.")

class ExperimentLogger:
    def __init__(self, project="OrangeFruitNet", run_name=None, use_wandb=True, save_dir="./logs"):
        """
        Initialize the logger.
        Args:
            project (str): WandB project name
            run_name (str): Optional custom run name
            use_wandb (bool): Enable wandb if available
            save_dir (str): Local save directory for logs
        """
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{self.timestamp}"
        self.save_dir = os.path.join(save_dir, self.run_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        if self.use_wandb:
            wandb.init(project=project, name=self.run_name, config={})
            print(f"[INFO] W&B logging initialized for run: {self.run_name}")
        else:
            print(f"[INFO] Local logging mode active: {self.save_dir}")

        self.metrics_file = os.path.join(self.save_dir, "metrics.json")
        self.logs = []

    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log training or evaluation metrics.
        Args:
            metrics (dict): Dictionary of metric_name → value
            step (int): Optional training step or epoch number
        """
        metrics["step"] = step
        metrics["timestamp"] = datetime.datetime.now().isoformat()
        self.logs.append(metrics)

        if self.use_wandb:
            wandb.log(metrics)
        else:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")

    def log_config(self, config: dict):
        """
        Save training configuration (hyperparameters).
        """
        cfg_path = os.path.join(self.save_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=4)
        if self.use_wandb:
            wandb.config.update(config)

    def save_model(self, model, filename="best_model.pt"):
        """
        Save model checkpoint (PyTorch-compatible).
        """
        import torch
        model_path = os.path.join(self.save_dir, filename)
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Model saved: {model_path}")
        if self.use_wandb:
            wandb.save(model_path)

    def close(self):
        """
        Finalize logging session.
        """
        if self.use_wandb:
            wandb.finish()
        print(f"[INFO] Logging session closed for run: {self.run_name}")
