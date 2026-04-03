"""
Supervised pretraining on a large source dataset (e.g., CASIA 3,891 Chinese character classes).
Trains only the ResNet50 backbone + a simple classification head.
The trained backbone weights can then be used to initialize fine-tuning on the target dataset (Ekush).
"""

import os
import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets.folder import (
    DatasetFolder,
    default_loader,
    IMG_EXTENSIONS,
    make_dataset,
)
from tqdm import tqdm
from torch.amp import autocast, GradScaler

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

from model import ResNet50Backbone
from dataset import AlbumentationsImageFolder, get_transforms
from utils import set_seed, save_checkpoint, load_resume, MetricsLogger, print_resource_usage


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FixedClassImageFolder(DatasetFolder):
    def __init__(self, root, class_to_idx, transform=None, target_transform=None):
        self.class_to_idx = class_to_idx
        super().__init__(
            root, loader=default_loader, extensions=IMG_EXTENSIONS,
            transform=transform, target_transform=target_transform,
        )
        self.samples = make_dataset(self.root, self.class_to_idx, self.extensions, None)
        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples
        self.classes = list(class_to_idx.keys())


class SupervisedPretrainModel(nn.Module):
    """ResNet50 backbone + global average pool + classification head."""

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ResNet50Backbone(pretrained_imagenet=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.backbone(x)     # (B, 2048, H, W)
        feat = self.avgpool(feat)    # (B, 2048, 1, 1)
        feat = feat.flatten(1)       # (B, 2048)
        return self.fc(feat)


def get_sorted_class_mapping(train_dir):
    class_names = sorted(os.listdir(train_dir))
    all_numeric = all(name.isdigit() for name in class_names)
    if all_numeric:
        class_names = sorted(class_names, key=lambda x: int(x))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    print(f"[INFO] {len(class_to_idx)} classes detected")
    return class_to_idx, idx_to_class, class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Pretraining on Source Dataset")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Dataset root with train/val/test folders")
    parser.add_argument("--checkpoint_dir", type=str, default=str(PROJECT_ROOT / "chkp" / "supervised_pretrain"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (64-96 for 8GB VRAM with 3891 classes)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train(args=None):
    if args is None:
        args = parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    resume_path = checkpoint_dir / "resume_checkpoint.pth"
    best_path = checkpoint_dir / "best_checkpoint.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Class mapping
    train_dir = os.path.join(data_root, "train")
    class_to_idx, idx_to_class, class_names = get_sorted_class_mapping(train_dir)
    num_classes = len(class_to_idx)

    with open(checkpoint_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class, "class_names": class_names}, f)

    # Data
    train_tf, val_tf = get_transforms(args.input_size)

    train_folder = FixedClassImageFolder(os.path.join(data_root, "train"), class_to_idx)
    val_folder = FixedClassImageFolder(os.path.join(data_root, "val"), class_to_idx)

    train_ds = AlbumentationsImageFolder(train_folder, transform=train_tf)
    val_ds = AlbumentationsImageFolder(val_folder, transform=val_tf)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Classes: {num_classes}")

    import platform
    is_windows = platform.system() == "Windows"
    eff_workers = min(args.num_workers, 4) if is_windows else args.num_workers

    train_kwargs = {"batch_size": args.batch_size, "shuffle": True,
                    "num_workers": eff_workers, "pin_memory": True}
    if eff_workers > 0:
        train_kwargs.update({"persistent_workers": True, "prefetch_factor": 3})
    train_loader = DataLoader(train_ds, **train_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=min(eff_workers, 2), pin_memory=True)

    # Model
    model = SupervisedPretrainModel(num_classes).to(device, memory_format=torch.channels_last)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    amp_enabled = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    # Resume
    best_acc = 0.0
    start_epoch = 0
    ckpt = load_resume(resume_path, device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    metrics_logger = MetricsLogger(checkpoint_dir / "training_metrics.json")

    if start_epoch >= args.epochs:
        print("Training already completed.")
        return

    print_resource_usage("INIT")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            _, preds = logits.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)
            pbar.set_postfix(loss=loss.item(), acc=total_correct / total_samples)

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(imgs)
                _, preds = logits.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        train_acc = total_correct / total_samples if total_samples > 0 else 0
        train_loss_avg = total_loss / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch+1}: val_acc={val_acc:.4f}, best_acc={best_acc:.4f}")
        print_resource_usage(f"E{epoch+1}")
        metrics_logger.log(epoch, train_loss=train_loss_avg, train_acc=train_acc,
                           val_acc=val_acc, best_acc=best_acc, lr=optimizer.param_groups[0]["lr"])

        ckpt_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_acc": best_acc,
            "num_classes": num_classes,
        }
        save_checkpoint(ckpt_data, resume_path)
        if is_best:
            save_checkpoint(ckpt_data, best_path)

    # Save backbone-only weights for downstream use
    backbone_state = model.backbone.state_dict()
    torch.save({"backbone_state": backbone_state, "best_acc": best_acc},
               checkpoint_dir / "backbone_weights.pth")
    print(f"Backbone weights saved to: {checkpoint_dir / 'backbone_weights.pth'}")


if __name__ == "__main__":
    train()
