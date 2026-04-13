import os
import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import (
    DatasetFolder,
    default_loader,
    IMG_EXTENSIONS,
    make_dataset,
)
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import numpy as np

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

from model import HybridModel, ResNet50Backbone
from dataset import AlbumentationsImageFolder, get_transforms
from utils import set_seed, save_checkpoint, load_resume, compute_topk, MetricsLogger, print_resource_usage


# -----------------------------
# Defaults (overridable via CLI)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# training stages
STAGE_A_EPOCHS = 6  # head + transformer
STAGE_B_EPOCHS = 194  # full fine-tune

# LR groups
BACKBONE_LR = 1e-5
TRANSFORMER_LR = 1e-4
HEAD_LR = 1e-4
WEIGHT_DECAY = 5e-5


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Model Training (Cross-Script Transfer Experiments)")
    parser.add_argument(
        "--data_root", type=str, default=str(PROJECT_ROOT / "data"),
        help="Dataset root containing train/val/test folders"
    )
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes (auto-detected from train folder if not set)")
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size (128 for 8GB VRAM)")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # Backbone initialization
    parser.add_argument(
        "--backbone_init", type=str, default="random",
        choices=["random", "imagenet", "ssl"],
        help="Backbone initialization: random, imagenet (pretrained), or ssl (load from --ssl_checkpoint)"
    )
    parser.add_argument("--ssl_checkpoint", type=str, default=None,
                        help="Path to SSL (BYOL) checkpoint for backbone init")

    # Label fraction for label-efficiency experiments
    parser.add_argument(
        "--label_fraction", type=float, default=1.0,
        help="Fraction of training labels to use (0.01 to 1.0). Val/test always use 100%%."
    )

    # Experiment naming
    parser.add_argument(
        "--experiment_name", type=str, default="default",
        help="Name for this experiment run (used for checkpoint/log subdirectory)"
    )

    # Checkpoint paths (auto-generated from experiment_name if not set)
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    parser.add_argument("--stage_a_epochs", type=int, default=STAGE_A_EPOCHS)
    parser.add_argument("--stage_b_epochs", type=int, default=STAGE_B_EPOCHS)
    parser.add_argument("--prefer_best_resume", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


set_seed(42)


class FixedClassImageFolder(DatasetFolder):
    """ImageFolder that uses a provided class_to_idx mapping for consistent labels across splits."""

    def __init__(self, root, class_to_idx, transform=None, target_transform=None):
        self.class_to_idx = class_to_idx
        super().__init__(
            root,
            loader=default_loader,
            extensions=IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
        )
        self.samples = make_dataset(self.root, self.class_to_idx, self.extensions, None)
        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples
        self.classes = list(class_to_idx.keys())


def get_sorted_class_mapping(train_dir):
    """
    Create a consistent class_to_idx mapping using NUMERICAL sorting for numeric folder names.
    This ensures classes like '1', '2', ..., '50' are mapped to indices 0, 1, ..., 49
    instead of alphabetical order (1, 10, 11, ..., 19, 2, 20, ...).
    """
    class_names = sorted(os.listdir(train_dir))

    # Check if all class names are numeric
    all_numeric = all(name.isdigit() for name in class_names)

    if all_numeric:
        # Sort numerically
        class_names = sorted(class_names, key=lambda x: int(x))
        print(f"[INFO] Detected numeric class names. Using numerical sorting.")
    else:
        # Keep alphabetical sorting
        print(f"[INFO] Using alphabetical sorting for class names.")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    print(f"[INFO] Class mapping created: {len(class_to_idx)} classes")
    print(f"[INFO] First 5: {list(class_to_idx.items())[:5]}")
    print(f"[INFO] Last 5: {list(class_to_idx.items())[-5:]}")

    return class_to_idx, idx_to_class, class_names


def save_class_mapping(class_to_idx, idx_to_class, class_names, path):
    """Save class mapping to JSON file."""
    mapping = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_names": class_names,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"[INFO] Class mapping saved to: {path}")


def load_class_mapping(path):
    """Load class mapping from JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # Convert idx_to_class keys back to int
    mapping["idx_to_class"] = {int(k): v for k, v in mapping["idx_to_class"].items()}
    return mapping


def extract_backbone_state(resnet, ckpt_state_dict):
    """Try to map keys from BYOL checkpoint to resnet state dict.
    Works by trying several common prefixes.
    """
    resnet_state = resnet.state_dict()
    mapped = {}
    candidates = list(ckpt_state_dict.keys())

    prefixes = ["backbone.", "backbone_momentum.", "module.backbone.", "module.", ""]

    for name in resnet_state.keys():
        found = False
        for p in prefixes:
            cand = p + name
            if cand in ckpt_state_dict:
                mapped[name] = ckpt_state_dict[cand]
                found = True
                break
        if not found:
            # try suffix match (last part)
            for k in candidates:
                if k.endswith(name):
                    mapped[name] = ckpt_state_dict[k]
                    found = True
                    break
    return mapped


def load_ssl_backbone(resnet, ssl_ckpt_path, device):
    if not os.path.exists(ssl_ckpt_path):
        print(f"[WARN] SSL checkpoint not found: {ssl_ckpt_path}")
        return resnet
    ckpt = torch.load(ssl_ckpt_path, map_location="cpu", weights_only=False)
    # ckpt contains: 'model_state'
    state = None
    if isinstance(ckpt, dict) and ("model_state" in ckpt or "state_dict" in ckpt):
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt

    mapped = extract_backbone_state(resnet, state)
    missing_keys = [k for k in resnet.state_dict().keys() if k not in mapped]
    if len(mapped) == 0:
        print(
            "[WARN] No backbone keys mapped automatically from SSL checkpoint. Loading with strict=False as fallback."
        )
        try:
            resnet.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"Fallback load failed: {e}")
        return resnet

    resnet_state = resnet.state_dict()
    resnet_state.update(mapped)
    resnet.load_state_dict(resnet_state)
    print("[INFO] SSL backbone weights loaded (partial/complete).")
    return resnet


def subsample_dataset(dataset, fraction, seed=42):
    """Return a Subset of dataset using stratified sampling by class label."""
    if fraction >= 1.0:
        return dataset
    rng = np.random.RandomState(seed)
    targets = np.array([dataset.image_folder.imgs[i][1] for i in range(len(dataset))])
    classes = np.unique(targets)
    indices = []
    for c in classes:
        class_indices = np.where(targets == c)[0]
        n_keep = max(1, int(len(class_indices) * fraction))
        chosen = rng.choice(class_indices, size=n_keep, replace=False)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    print(f"[INFO] Label fraction={fraction:.2f}: using {len(indices)}/{len(dataset)} training samples")
    return Subset(dataset, indices)


def build_data_loaders(root, input_size, batch_size, num_workers, class_to_idx,
                       label_fraction=1.0, eval_batch_size=128, seed=42):
    """Build data loaders using a fixed class_to_idx mapping for all splits."""
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")

    train_tf, val_tf = get_transforms(input_size)

    train_folder = FixedClassImageFolder(train_dir, class_to_idx)
    val_folder = FixedClassImageFolder(val_dir, class_to_idx)
    test_folder = FixedClassImageFolder(test_dir, class_to_idx)

    train_ds = AlbumentationsImageFolder(train_folder, transform=train_tf)
    val_ds = AlbumentationsImageFolder(val_folder, transform=val_tf)
    test_ds = AlbumentationsImageFolder(test_folder, transform=val_tf)

    # Apply label fraction (only to training data)
    if label_fraction < 1.0:
        train_ds = subsample_dataset(train_ds, label_fraction, seed=seed)

    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Val samples: {len(val_ds)}")
    print(f"[INFO] Test samples: {len(test_ds)}")

    import platform
    is_windows = platform.system() == "Windows"
    # On Windows: cap at 2 workers, persistent_workers=False to avoid page file exhaustion
    effective_workers = min(num_workers, 2) if is_windows else num_workers

    train_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": effective_workers,
        "pin_memory": torch.cuda.is_available(),
        "shuffle": True,
    }
    if effective_workers > 0:
        train_loader_kwargs.update({"persistent_workers": False, "prefetch_factor": 2})

    eval_loader_kwargs = {
        "batch_size": eval_batch_size,
        "num_workers": min(effective_workers, 2),
        "pin_memory": torch.cuda.is_available(),
        "shuffle": False,
    }

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **eval_loader_kwargs)
    test_loader = DataLoader(test_ds, **eval_loader_kwargs)

    return train_loader, val_loader, test_loader


def build_optimizer(model, stage: str):
    """Build optimizer for a given training stage."""
    if stage == "A":
        return optim.AdamW(
            [
                {"params": model.head.parameters(), "lr": TRANSFORMER_LR},
                {"params": model.embedding_fc.parameters(), "lr": HEAD_LR},
                {"params": model.arcface.parameters(), "lr": HEAD_LR},
            ],
            weight_decay=WEIGHT_DECAY,
        )

    if stage == "B":
        return optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
                {"params": model.head.parameters(), "lr": TRANSFORMER_LR},
                {"params": model.embedding_fc.parameters(), "lr": HEAD_LR},
                {"params": model.arcface.parameters(), "lr": HEAD_LR},
            ],
            weight_decay=WEIGHT_DECAY,
        )

    raise ValueError(f"Unknown stage: {stage}")


def set_backbone_trainable(model, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def pick_resume_checkpoint(device, best_path, resume_path, prefer_best=True):
    """Return checkpoint dict and source label ('best' or 'last')."""
    source = None
    ckpt = None
    if prefer_best:
        ckpt = load_resume(best_path, device)
        if ckpt is not None:
            source = "best"
    if ckpt is None:
        ckpt = load_resume(resume_path, device)
        if ckpt is not None:
            source = "last"
    return ckpt, source


def train(args=None):
    if args is None:
        args = parse_args()

    set_seed(args.seed)

    DATA_ROOT = Path(args.data_root)
    INPUT_SIZE = args.input_size
    BATCH_SIZE = args.batch_size
    EVAL_BATCH_SIZE = args.eval_batch_size
    NUM_WORKERS = args.num_workers
    D_MODEL = 512
    TRANSFORMER_LAYERS = 4
    BACKBONE_CHANNELS = 2048

    # Checkpoint paths based on experiment name
    CHECKPOINT_DIR = Path(args.checkpoint_dir) if args.checkpoint_dir else PROJECT_ROOT / "chkp" / args.experiment_name
    RESUME_CHECKPOINT = CHECKPOINT_DIR / "resume_checkpoint.pth"
    BEST_CHECKPOINT = CHECKPOINT_DIR / "best_checkpoint.pth"
    CLASS_MAPPING_FILE = CHECKPOINT_DIR / "class_mapping.json"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    metrics_logger = MetricsLogger(CHECKPOINT_DIR / "training_metrics.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Experiment: {args.experiment_name}")
    print(f"Backbone init: {args.backbone_init}")
    print(f"Label fraction: {args.label_fraction}")
    print(f"Data root: {DATA_ROOT}")
    print_resource_usage("INIT")

    # Create or load consistent class mapping
    train_dir = os.path.join(DATA_ROOT, "train")
    class_to_idx, idx_to_class, class_names = get_sorted_class_mapping(train_dir)
    save_class_mapping(class_to_idx, idx_to_class, class_names, CLASS_MAPPING_FILE)

    NUM_CLASSES = args.num_classes if args.num_classes else len(class_to_idx)
    assert (
        len(class_to_idx) == NUM_CLASSES
    ), f"NUM_CLASSES ({NUM_CLASSES}) != detected classes ({len(class_to_idx)})"

    # data loaders with fixed class mapping + label fraction
    train_loader, val_loader, test_loader = build_data_loaders(
        DATA_ROOT, INPUT_SIZE, BATCH_SIZE, NUM_WORKERS, class_to_idx,
        label_fraction=args.label_fraction, eval_batch_size=EVAL_BATCH_SIZE,
        seed=args.seed,
    )

    # Model backbone initialization
    if args.backbone_init == "imagenet":
        resnet_backbone = ResNet50Backbone(pretrained_imagenet=True)
    elif args.backbone_init == "ssl" and args.ssl_checkpoint:
        resnet_backbone = ResNet50Backbone(pretrained_imagenet=False)
        resnet_backbone = load_ssl_backbone(resnet_backbone, args.ssl_checkpoint, device)
    else:
        resnet_backbone = ResNet50Backbone(pretrained_imagenet=False)
        if args.backbone_init == "ssl":
            print("[WARN] --backbone_init=ssl but no --ssl_checkpoint provided. Using random init.")

    model = HybridModel(
        num_classes=NUM_CLASSES,
        backbone=resnet_backbone,
        d_model=D_MODEL,
        transformer_layers=TRANSFORMER_LAYERS,
        backbone_channels=BACKBONE_CHANNELS,
        input_size=INPUT_SIZE,
    )
    model = model.to(device, memory_format=torch.channels_last)

    amp_enabled = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    start_epoch = 0
    start_stage = "A"

    resume, resume_source = pick_resume_checkpoint(
        device, BEST_CHECKPOINT, RESUME_CHECKPOINT,
        prefer_best=args.prefer_best_resume,
    )
    if resume is not None:
        ckpt_epoch = int(resume.get("epoch", 0))
        start_epoch = ckpt_epoch + 1
        best_acc = float(resume.get("best_acc", 0.0))
        start_stage = resume.get("stage", "A")
        if start_stage not in ("A", "B"):
            start_stage = "A" if ckpt_epoch < args.stage_a_epochs else "B"

        print(
            f"[INFO] Resume checkpoint found ({resume_source}) "
            f"(epoch={ckpt_epoch}, stage={start_stage}, best_acc={best_acc:.4f})"
        )
        model.load_state_dict(resume["model_state"])
        if "scaler_state" in resume:
            scaler.load_state_dict(resume["scaler_state"])
    else:
        print("[INFO] No resume checkpoint found. Training from scratch.")

    total_epochs = args.stage_a_epochs + args.stage_b_epochs

    # Check if training is already complete
    if start_epoch >= total_epochs:
        print(
            f"[INFO] Training already completed (start_epoch={start_epoch} >= total_epochs={total_epochs}). "
            "Running final test evaluation only."
        )
        best_ckpt = load_resume(BEST_CHECKPOINT, device)
        if best_ckpt is not None:
            model.load_state_dict(best_ckpt["model_state"])
        test_acc = validate(
            model, test_loader, device, class_names, split_name="test(best)"
        )
        print(f"Final test accuracy: {test_acc:.4f}")
        return

    # Determine if we should skip Stage A (resuming in Stage B)
    skip_stage_a = (start_epoch >= args.stage_a_epochs) or (
        start_stage == "B" and start_epoch >= args.stage_a_epochs
    )

    # Stage A setup (backbone frozen)
    set_backbone_trainable(model, trainable=False)
    optimizer = build_optimizer(model, stage="A")
    scheduler = None

    if resume is not None and start_stage == "A" and not skip_stage_a:
        if "optimizer_state" in resume:
            try:
                optimizer.load_state_dict(resume["optimizer_state"])
                print("[INFO] Stage A optimizer state restored.")
            except Exception as e:
                print(f"[WARN] Could not restore optimizer state: {e}")

    if not skip_stage_a:
        for epoch in range(start_epoch, args.stage_a_epochs):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            pbar = tqdm(train_loader, desc=f"StageA Epoch {epoch+1}/{args.stage_a_epochs}")
            for imgs, labels in pbar:
                imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(imgs, labels)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * imgs.size(0)
                _, preds = logits.max(1)
                total_correct += (preds == labels).sum().item()
                total_samples += imgs.size(0)
                pbar.set_postfix(loss=loss.item(), acc=total_correct / total_samples)

            train_acc = total_correct / total_samples if total_samples > 0 else 0
            train_loss = total_loss / total_samples if total_samples > 0 else 0
            val_acc = validate(model, val_loader, device, class_names)
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            res = print_resource_usage(f"A-E{epoch+1}")
            metrics_logger.log(epoch, stage="A", train_loss=train_loss, train_acc=train_acc,
                               val_acc=val_acc, best_acc=best_acc, lr=optimizer.param_groups[0]["lr"])

            ckpt_data = {
                "epoch": epoch,
                "stage": "A",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc": best_acc,
                "scaler_state": scaler.state_dict(),
                "class_to_idx": class_to_idx,
                "experiment": args.experiment_name,
                "backbone_init": args.backbone_init,
                "label_fraction": args.label_fraction,
            }
            save_checkpoint(ckpt_data, RESUME_CHECKPOINT)
            if is_best:
                save_checkpoint(ckpt_data, BEST_CHECKPOINT)

    # Clean up data loaders before Stage B to free shared memory (Windows fix)
    del train_loader, val_loader, test_loader
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Rebuild data loaders for Stage B
    train_loader, val_loader, test_loader = build_data_loaders(
        DATA_ROOT, INPUT_SIZE, BATCH_SIZE, NUM_WORKERS, class_to_idx,
        label_fraction=args.label_fraction, eval_batch_size=EVAL_BATCH_SIZE,
        seed=args.seed,
    )
    print("[INFO] Data loaders rebuilt for Stage B")

    set_backbone_trainable(model, trainable=True)
    optimizer = build_optimizer(model, stage="B")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage_b_epochs
    )

    if resume is not None and (start_stage == "B" or skip_stage_a):
        if "optimizer_state" in resume:
            try:
                optimizer.load_state_dict(resume["optimizer_state"])
                print("[INFO] Stage B optimizer state restored.")
            except Exception as e:
                print(f"[WARN] Could not restore Stage B optimizer state: {e}")
        if "scheduler_state" in resume:
            try:
                scheduler.load_state_dict(resume["scheduler_state"])
                print("[INFO] Stage B scheduler state restored.")
            except Exception as e:
                print(f"[WARN] Could not restore scheduler state: {e}")

    stage_b_start = max(start_epoch, args.stage_a_epochs) if skip_stage_a else args.stage_a_epochs

    for epoch in range(stage_b_start, args.stage_a_epochs + args.stage_b_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        pbar = tqdm(
            train_loader, desc=f"StageB Epoch {epoch+1}/{args.stage_a_epochs+args.stage_b_epochs}"
        )
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(imgs, labels)
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

        train_acc = total_correct / total_samples if total_samples > 0 else 0
        train_loss = total_loss / total_samples if total_samples > 0 else 0
        val_acc = validate(model, val_loader, device, class_names)
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        res = print_resource_usage(f"B-E{epoch+1}")
        metrics_logger.log(epoch, stage="B", train_loss=train_loss, train_acc=train_acc,
                           val_acc=val_acc, best_acc=best_acc, lr=optimizer.param_groups[0]["lr"])

        ckpt_data = {
            "epoch": epoch,
            "stage": "B",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_acc,
            "scaler_state": scaler.state_dict(),
            "class_to_idx": class_to_idx,
            "experiment": args.experiment_name,
            "backbone_init": args.backbone_init,
            "label_fraction": args.label_fraction,
        }
        save_checkpoint(ckpt_data, RESUME_CHECKPOINT)
        if is_best:
            save_checkpoint(ckpt_data, BEST_CHECKPOINT)

    # Final evaluation on test set
    best_ckpt = load_resume(BEST_CHECKPOINT, device)
    if best_ckpt is None:
        print(f"[WARN] Best checkpoint not found. Falling back to last weights.")
        test_acc = validate(
            model, test_loader, device, class_names, split_name="test(last)"
        )
    else:
        model.load_state_dict(best_ckpt["model_state"])
        test_acc = validate(
            model, test_loader, device, class_names, split_name="test(best)"
        )
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Results saved to: {CHECKPOINT_DIR}")


@torch.no_grad()
def validate(model, loader, device, class_names=None, split_name: str = "val"):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        # Strict-protocol eval: label-free inference (no GT labels used inside the model)
        emb = model(imgs, labels=None)
        logits = model.arcface(emb, labels=None)
        _, preds = logits.max(1)
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"Total samples in {split_name}: {total_samples}")
    print(f"{split_name.capitalize()} accuracy: {acc:.4f}")

    # Print class distribution check
    if class_names is not None:
        import numpy as np

        unique_labels, counts = np.unique(all_labels, return_counts=True)
        print(
            f"[DEBUG] Unique labels in {split_name}: {len(unique_labels)}, range: [{min(all_labels)}, {max(all_labels)}]"
        )
        unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
        print(
            f"[DEBUG] Unique predictions: {len(unique_preds)}, range: [{min(all_preds)}, {max(all_preds)}]"
        )

    return acc


if __name__ == "__main__":
    train()
