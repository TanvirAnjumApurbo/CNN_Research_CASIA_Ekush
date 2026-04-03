import os
import json
import time
import torch
import random
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_resume(path, device):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def compute_topk(output, target, k=1):
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return correct[:k].reshape(-1).float().sum(0, keepdim=True).item()


# ============================================================
# GPU / RAM MONITORING
# ============================================================

def get_resource_usage():
    """Return dict with GPU VRAM and system RAM usage."""
    info = {}
    if torch.cuda.is_available():
        info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        info["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_used_mb"] = mem.used / 1024**2
        info["ram_total_mb"] = mem.total / 1024**2
        info["ram_percent"] = mem.percent
    except ImportError:
        pass
    return info


def print_resource_usage(prefix=""):
    """Print current GPU VRAM and system RAM usage."""
    info = get_resource_usage()
    parts = []
    if "gpu_allocated_mb" in info:
        parts.append(f"VRAM: {info['gpu_allocated_mb']:.0f}/{info['gpu_reserved_mb']:.0f}MB "
                     f"(peak {info['gpu_max_allocated_mb']:.0f}MB)")
    if "ram_used_mb" in info:
        parts.append(f"RAM: {info['ram_used_mb']:.0f}/{info['ram_total_mb']:.0f}MB ({info['ram_percent']:.0f}%)")
    if parts:
        tag = f"[{prefix}] " if prefix else ""
        print(f"{tag}Resources: {' | '.join(parts)}")
    return info


# ============================================================
# TRAINING METRICS LOGGER
# ============================================================

class MetricsLogger:
    """Saves per-epoch metrics to a JSON file for later plotting."""

    def __init__(self, save_path):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.history = []
        # Load existing history if resuming
        if self.save_path.exists():
            try:
                with open(self.save_path, "r") as f:
                    self.history = json.load(f)
            except (json.JSONDecodeError, Exception):
                self.history = []

    def log(self, epoch, **kwargs):
        """Log metrics for an epoch. kwargs: train_loss, train_acc, val_acc, lr, etc."""
        entry = {"epoch": epoch, "timestamp": time.time()}
        entry.update(kwargs)
        # Add resource usage
        entry.update(get_resource_usage())
        # Replace existing epoch entry or append
        existing = [i for i, e in enumerate(self.history) if e.get("epoch") == epoch]
        if existing:
            self.history[existing[0]] = entry
        else:
            self.history.append(entry)
        self._save()

    def _save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_history(self):
        return self.history
