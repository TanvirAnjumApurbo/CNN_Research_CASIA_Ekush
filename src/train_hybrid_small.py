import os
import math
import argparse
from pathlib import Path
import json
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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
from utils import set_seed, save_checkpoint, load_resume, MetricsLogger, print_resource_usage


# -----------------------------
# Defaults (overridable via CLI)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# training stages
STAGE_A_EPOCHS = 6  # head + transformer
STAGE_B_EPOCHS = 94  # full fine-tune

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
                        help="Path to SSL/Supervised-pretrain checkpoint for backbone init")
    parser.add_argument(
        "--min_backbone_key_match_ratio", type=float, default=0.8,
        help="Minimum mapped key ratio required when loading backbone weights."
    )

    # Label fraction for label-efficiency experiments
    parser.add_argument(
        "--label_fraction", type=float, default=1.0,
        help="Fraction of training labels to use (0.01 to 1.0). Val/test always use 100%%."
    )

    # Step-budget methodology
    parser.add_argument(
        "--step_budget_mode", type=str, default="fixed",
        choices=["fixed", "epoch"],
        help="fixed: keep step budget comparable across fractions; epoch: legacy epoch-based budget."
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Override total train steps across Stage A+B. If unset and mode=fixed, derives from reference fraction."
    )
    parser.add_argument(
        "--step_budget_reference_fraction", type=float, default=0.10,
        help="Reference fraction for auto step budget in fixed mode."
    )

    # Precision controls
    parser.add_argument(
        "--stage_a_precision", type=str, default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Precision policy for Stage A."
    )
    parser.add_argument(
        "--stage_b_precision", type=str, default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Precision policy for Stage B. auto => bf16 on CUDA if supported, else fp32."
    )

    # Recovery controls
    parser.add_argument(
        "--recovery_lr_factor", type=float, default=0.5,
        help="Multiply all LR groups by this factor when non-finite parameter recovery is triggered."
    )
    parser.add_argument(
        "--max_recoveries_per_stage", type=int, default=3,
        help="Fail training if non-finite recovery exceeds this limit per stage."
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
    parser.add_argument("--prefer_best_resume", action="store_true", default=False)
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
    """Create a consistent class_to_idx mapping, numerical sort when folders are numeric."""
    class_names = sorted(os.listdir(train_dir))

    all_numeric = all(name.isdigit() for name in class_names)
    if all_numeric:
        class_names = sorted(class_names, key=lambda x: int(x))
        print("[INFO] Detected numeric class names. Using numerical sorting.")
    else:
        print("[INFO] Using alphabetical sorting for class names.")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    print(f"[INFO] Class mapping created: {len(class_to_idx)} classes")
    print(f"[INFO] First 5: {list(class_to_idx.items())[:5]}")
    print(f"[INFO] Last 5: {list(class_to_idx.items())[-5:]}")
    return class_to_idx, idx_to_class, class_names


def save_class_mapping(class_to_idx, idx_to_class, class_names, path):
    mapping = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_names": class_names,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"[INFO] Class mapping saved to: {path}")


def extract_backbone_state(resnet, ckpt_state_dict):
    """Map keys from BYOL/supervised checkpoint to backbone state dict."""
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
            for k in candidates:
                if k.endswith(name):
                    mapped[name] = ckpt_state_dict[k]
                    found = True
                    break
    return mapped


def load_ssl_backbone(resnet, ssl_ckpt_path, min_match_ratio=0.8):
    """Load backbone from SSL or supervised-pretrain checkpoint with strict mapping validation."""
    if not os.path.exists(ssl_ckpt_path):
        raise FileNotFoundError(f"SSL/Supervised checkpoint not found: {ssl_ckpt_path}")

    ckpt = torch.load(ssl_ckpt_path, map_location="cpu", weights_only=False)

    # Supported formats:
    # 1) {"model_state": ...} or {"state_dict": ...} from SSL
    # 2) {"backbone_state": ...} from supervised_pretrain.py
    # 3) raw state dict
    if isinstance(ckpt, dict):
        if "backbone_state" in ckpt and isinstance(ckpt["backbone_state"], dict):
            state = ckpt["backbone_state"]
            print("[INFO] Loading checkpoint format: backbone_state")
        elif "model_state" in ckpt or "state_dict" in ckpt:
            state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
            print("[INFO] Loading checkpoint format: model_state/state_dict")
        else:
            state = ckpt
            print("[INFO] Loading checkpoint format: raw dict state")
    else:
        state = ckpt
        print("[INFO] Loading checkpoint format: raw state object")

    if not isinstance(state, dict):
        raise RuntimeError("Unsupported checkpoint payload: expected a state dict.")

    mapped = extract_backbone_state(resnet, state)
    total_backbone_keys = len(resnet.state_dict())
    mapped_ratio = len(mapped) / max(1, total_backbone_keys)

    print(
        f"[INFO] Backbone key mapping: {len(mapped)}/{total_backbone_keys} "
        f"({mapped_ratio * 100:.1f}%)"
    )
    if mapped_ratio < min_match_ratio:
        sample_keys = list(state.keys())[:10]
        raise RuntimeError(
            f"Backbone mapping ratio too low ({mapped_ratio:.3f} < {min_match_ratio:.3f}). "
            f"Likely wrong checkpoint format/path. Sample keys: {sample_keys}"
        )

    resnet_state = resnet.state_dict()
    resnet_state.update(mapped)
    resnet.load_state_dict(resnet_state)
    print("[INFO] Backbone weights loaded successfully.")
    return resnet


def subsample_dataset(dataset, fraction, seed=42):
    """Return stratified Subset by class label."""
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
    """Build data loaders and return full-train sample count."""
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")

    train_tf, val_tf = get_transforms(input_size)

    train_folder = FixedClassImageFolder(train_dir, class_to_idx)
    val_folder = FixedClassImageFolder(val_dir, class_to_idx)
    test_folder = FixedClassImageFolder(test_dir, class_to_idx)

    full_train_samples = len(train_folder)

    train_ds = AlbumentationsImageFolder(train_folder, transform=train_tf)
    val_ds = AlbumentationsImageFolder(val_folder, transform=val_tf)
    test_ds = AlbumentationsImageFolder(test_folder, transform=val_tf)

    if label_fraction < 1.0:
        train_ds = subsample_dataset(train_ds, label_fraction, seed=seed)

    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Val samples: {len(val_ds)}")
    print(f"[INFO] Test samples: {len(test_ds)}")

    import platform
    is_windows = platform.system() == "Windows"
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

    return train_loader, val_loader, test_loader, full_train_samples


def build_optimizer(model, stage: str, backbone_init: str = "random"):
    if stage == "A":
        return optim.AdamW(
            [
                {"params": model.head.parameters(), "lr": TRANSFORMER_LR},
                {"params": model.embedding_fc.parameters(), "lr": HEAD_LR},
                {"params": model.classifier.parameters(), "lr": HEAD_LR},
            ],
            weight_decay=WEIGHT_DECAY,
        )

    if stage == "B":
        bb_lr = TRANSFORMER_LR if backbone_init == "random" else BACKBONE_LR
        return optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": bb_lr},
                {"params": model.head.parameters(), "lr": TRANSFORMER_LR},
                {"params": model.embedding_fc.parameters(), "lr": HEAD_LR},
                {"params": model.classifier.parameters(), "lr": HEAD_LR},
            ],
            weight_decay=WEIGHT_DECAY,
        )

    raise ValueError(f"Unknown stage: {stage}")


def set_backbone_trainable(model, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def pick_resume_checkpoint(device, best_path, resume_path, prefer_best=True):
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


def resolve_precision_mode(requested_mode, stage, device):
    """Resolve auto precision with a safety-first policy."""
    if device.type != "cuda":
        return "fp32"

    if requested_mode != "auto":
        if requested_mode == "bf16" and not torch.cuda.is_bf16_supported():
            print("[WARN] bf16 requested but not supported on this GPU. Falling back to fp32.")
            return "fp32"
        return requested_mode

    if stage == "A":
        # Stage A is usually stable and can run fast in fp16.
        return "fp16"

    # Stage B (critical): prefer bf16, otherwise run full fp32.
    return "bf16" if torch.cuda.is_bf16_supported() else "fp32"


def downgrade_precision_mode(current_mode, device):
    """Move to a safer precision after non-finite parameter event."""
    if current_mode == "fp16":
        return "bf16" if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else "fp32"
    if current_mode == "bf16":
        return "fp32"
    return "fp32"


def autocast_kwargs(device, precision_mode):
    if device.type != "cuda" or precision_mode == "fp32":
        return {"enabled": False}
    if precision_mode == "bf16":
        return {"enabled": True, "dtype": torch.bfloat16}
    # fp16
    return {"enabled": True, "dtype": torch.float16}


def build_grad_scaler(precision_mode, num_train_batches):
    if precision_mode != "fp16":
        return None
    num_train_batches = max(1, int(num_train_batches))
    return GradScaler(
        "cuda",
        enabled=True,
        init_scale=2**10,
        growth_interval=num_train_batches,
        growth_factor=1.5,
        backoff_factor=0.5,
    )


def get_scaler_scale_str(scaler):
    if scaler is None:
        return "N/A"
    try:
        return f"{scaler.get_scale():.0f}"
    except Exception:
        return "N/A"


def has_nonfinite_gradients(model):
    for p in model.parameters():
        if p.grad is not None and p.grad.dtype.is_floating_point and not torch.isfinite(p.grad).all():
            return True
    return False


def has_nonfinite_parameters(model):
    for p in model.parameters():
        if p.dtype.is_floating_point and not torch.isfinite(p).all():
            return True
    return False


def state_dict_has_nonfinite(state_dict):
    for value in state_dict.values():
        if torch.is_tensor(value) and value.dtype.is_floating_point and not torch.isfinite(value).all():
            return True
    return False


def optimizer_state_has_nonfinite(opt_state):
    if not isinstance(opt_state, dict):
        return False
    state = opt_state.get("state", {})
    for entry in state.values():
        if not isinstance(entry, dict):
            continue
        for value in entry.values():
            if torch.is_tensor(value) and value.dtype.is_floating_point and not torch.isfinite(value).all():
                return True
    return False


def reduce_optimizer_lrs(optimizer, factor):
    factor = float(factor)
    if factor <= 0.0 or factor >= 1.0:
        return
    for group in optimizer.param_groups:
        group["lr"] = max(1e-8, group["lr"] * factor)


def recover_from_last_finite_checkpoint(
    model, optimizer, scheduler, scaler, device, resume_path, best_path, stage
):
    """
    Restore last finite checkpoint (prefer resume, fallback best).
    Returns (ckpt, source_label, restored_scaler_state_flag).
    """
    candidates = [("last", resume_path), ("best", best_path)]
    last_error = None
    for source, path in candidates:
        ckpt = load_resume(path, device)
        if ckpt is None or "model_state" not in ckpt:
            continue
        if state_dict_has_nonfinite(ckpt["model_state"]):
            print(f"[WARN] Skipping {source} checkpoint recovery: model_state has non-finite values.")
            continue
        if "optimizer_state" in ckpt and optimizer_state_has_nonfinite(ckpt["optimizer_state"]):
            print(f"[WARN] Skipping {source} checkpoint recovery: optimizer_state has non-finite values.")
            continue
        try:
            model.load_state_dict(ckpt["model_state"])
            if optimizer is not None and "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if scheduler is not None and "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])

            scaler_restored = False
            if scaler is not None and "scaler_state" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state"])
                scaler_restored = True

            print(f"[INFO] Recovered training state from {source} checkpoint.")
            return ckpt, source, scaler_restored
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Failed to recover from finite checkpoints. Last error: {last_error}")


def compute_step_budgets(args, full_train_samples, train_loader_len):
    total_epochs = args.stage_a_epochs + args.stage_b_epochs
    if total_epochs <= 0:
        return 0, 0, 0

    if args.step_budget_mode == "fixed":
        if args.max_steps is not None:
            total_steps = max(1, int(args.max_steps))
        else:
            ref_frac = min(max(args.step_budget_reference_fraction, 1e-6), 1.0)
            ref_samples = max(1, int(full_train_samples * ref_frac))
            ref_batches = max(1, math.ceil(ref_samples / args.batch_size))
            total_steps = ref_batches * total_epochs
    else:
        # legacy behavior
        total_steps = max(1, train_loader_len * total_epochs)

    stage_a_ratio = args.stage_a_epochs / total_epochs
    stage_a_steps = int(round(total_steps * stage_a_ratio))
    stage_b_steps = total_steps - stage_a_steps

    if args.stage_a_epochs > 0 and stage_a_steps == 0:
        stage_a_steps = 1
        stage_b_steps = max(0, total_steps - stage_a_steps)
    if args.stage_b_epochs > 0 and stage_b_steps == 0:
        stage_b_steps = 1
        stage_a_steps = max(0, total_steps - stage_b_steps)

    return total_steps, stage_a_steps, stage_b_steps


def infer_step_progress_from_legacy_ckpt(ckpt_epoch, ckpt_stage, train_loader_len, stage_a_epochs, stage_a_budget, stage_b_budget):
    """Best-effort inference for older checkpoints that have epoch-only progress."""
    completed_epochs = ckpt_epoch + 1
    if ckpt_stage == "A":
        stage_a_done = min(stage_a_budget, completed_epochs * train_loader_len)
        stage_b_done = 0
    else:
        stage_a_done = stage_a_budget
        completed_b_epochs = max(0, completed_epochs - stage_a_epochs)
        stage_b_done = min(stage_b_budget, completed_b_epochs * train_loader_len)
    return stage_a_done, stage_b_done


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    precision_mode,
    scaler,
    stage_name,
    epoch_idx,
    epoch_total_est,
    steps_done,
    step_budget,
    scheduler=None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nonfinite_grad_batches = 0
    param_event = False

    pbar = tqdm(loader, desc=f"{stage_name} Epoch {epoch_idx}/{epoch_total_est}")
    ac_kwargs = autocast_kwargs(device, precision_mode)

    for imgs, labels in pbar:
        if steps_done >= step_budget:
            break

        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, **ac_kwargs):
            logits = model(imgs)
            loss = criterion(logits, labels)

        loss_val = float(loss.item())
        if not math.isfinite(loss_val):
            print(f"[WARN] Non-finite loss ({loss_val}) detected, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        grad_nonfinite = has_nonfinite_gradients(model)
        if grad_nonfinite:
            nonfinite_grad_batches += 1
            if scaler is not None:
                # GradScaler will skip optimizer step on inf grads and back off scale.
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(
                loss=loss_val,
                acc=(total_correct / total_samples if total_samples > 0 else 0.0),
                grad_nf=nonfinite_grad_batches,
                scale=get_scaler_scale_str(scaler),
            )
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Per-step parameter finite check (critical safety gate)
        if has_nonfinite_parameters(model):
            print(
                f"[CRIT] Non-finite parameter detected after optimizer step "
                f"(stage={stage_name}, epoch={epoch_idx})."
            )
            param_event = True
            break

        total_loss += loss_val * imgs.size(0)
        _, preds = logits.max(1)
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)
        steps_done += 1

        pbar.set_postfix(
            loss=loss_val,
            acc=(total_correct / total_samples if total_samples > 0 else 0.0),
            grad_nf=nonfinite_grad_batches,
            scale=get_scaler_scale_str(scaler),
        )

    train_acc = total_correct / total_samples if total_samples > 0 else 0.0
    train_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "total_samples": total_samples,
        "steps_done": steps_done,
        "nonfinite_grad_batches": nonfinite_grad_batches,
        "nonfinite_param_event": param_event,
    }


def train(args=None):
    if args is None:
        args = parse_args()

    set_seed(args.seed)

    # Random backbone: skip Stage A.
    if args.backbone_init == "random":
        total = args.stage_a_epochs + args.stage_b_epochs
        args.stage_a_epochs = 0
        args.stage_b_epochs = total

    DATA_ROOT = Path(args.data_root)
    INPUT_SIZE = args.input_size
    BATCH_SIZE = args.batch_size
    EVAL_BATCH_SIZE = args.eval_batch_size
    NUM_WORKERS = args.num_workers
    D_MODEL = 512
    TRANSFORMER_LAYERS = 4
    BACKBONE_CHANNELS = 2048

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

    train_dir = os.path.join(DATA_ROOT, "train")
    class_to_idx, idx_to_class, class_names = get_sorted_class_mapping(train_dir)
    save_class_mapping(class_to_idx, idx_to_class, class_names, CLASS_MAPPING_FILE)

    NUM_CLASSES = args.num_classes if args.num_classes else len(class_to_idx)
    assert len(class_to_idx) == NUM_CLASSES, (
        f"NUM_CLASSES ({NUM_CLASSES}) != detected classes ({len(class_to_idx)})"
    )

    train_loader, val_loader, test_loader, full_train_samples = build_data_loaders(
        DATA_ROOT, INPUT_SIZE, BATCH_SIZE, NUM_WORKERS, class_to_idx,
        label_fraction=args.label_fraction, eval_batch_size=EVAL_BATCH_SIZE, seed=args.seed,
    )

    total_step_budget, stage_a_step_budget, stage_b_step_budget = compute_step_budgets(
        args, full_train_samples=full_train_samples, train_loader_len=len(train_loader)
    )
    print(
        f"[INFO] Step budget mode={args.step_budget_mode} "
        f"(total={total_step_budget}, stageA={stage_a_step_budget}, stageB={stage_b_step_budget})"
    )

    if args.backbone_init == "imagenet":
        resnet_backbone = ResNet50Backbone(pretrained_imagenet=True)
    elif args.backbone_init == "ssl" and args.ssl_checkpoint:
        resnet_backbone = ResNet50Backbone(pretrained_imagenet=False)
        resnet_backbone = load_ssl_backbone(
            resnet_backbone, args.ssl_checkpoint, min_match_ratio=args.min_backbone_key_match_ratio
        )
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
    ).to(device, memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()

    stage_a_precision = resolve_precision_mode(args.stage_a_precision, "A", device)
    stage_b_precision = resolve_precision_mode(args.stage_b_precision, "B", device)
    print(f"[INFO] Precision policy: StageA={stage_a_precision}, StageB={stage_b_precision}")

    best_acc = 0.0
    logical_epoch = 0
    stage_a_steps_done = 0
    stage_b_steps_done = 0
    global_step = 0
    start_stage = "A"
    training_complete = False

    resume, resume_source = pick_resume_checkpoint(
        device, BEST_CHECKPOINT, RESUME_CHECKPOINT, prefer_best=args.prefer_best_resume
    )
    if resume is not None:
        ckpt_epoch = int(resume.get("epoch", 0))
        logical_epoch = ckpt_epoch + 1
        best_acc = float(resume.get("best_acc", 0.0))
        start_stage = resume.get("stage", "A")
        if start_stage not in ("A", "B"):
            start_stage = "A"

        model.load_state_dict(resume["model_state"])
        if has_nonfinite_parameters(model):
            raise RuntimeError(
                f"Resume checkpoint ({resume_source}) has non-finite parameters. "
                "Delete/replace corrupted resume checkpoint before continuing."
            )

        if "stage_a_steps_done" in resume and "stage_b_steps_done" in resume:
            stage_a_steps_done = int(resume.get("stage_a_steps_done", 0))
            stage_b_steps_done = int(resume.get("stage_b_steps_done", 0))
        else:
            inferred_a, inferred_b = infer_step_progress_from_legacy_ckpt(
                ckpt_epoch=ckpt_epoch,
                ckpt_stage=start_stage,
                train_loader_len=len(train_loader),
                stage_a_epochs=args.stage_a_epochs,
                stage_a_budget=stage_a_step_budget,
                stage_b_budget=stage_b_step_budget,
            )
            stage_a_steps_done = inferred_a
            stage_b_steps_done = inferred_b
            print("[WARN] Legacy checkpoint without step counters; using inferred progress.")

        global_step = int(resume.get("global_step", stage_a_steps_done + stage_b_steps_done))
        training_complete = bool(resume.get("training_complete", False))
        print(
            f"[INFO] Resume checkpoint found ({resume_source}) "
            f"(epoch={ckpt_epoch}, stage={start_stage}, best_acc={best_acc:.4f}, "
            f"stageA_steps={stage_a_steps_done}, stageB_steps={stage_b_steps_done})"
        )
    else:
        print("[INFO] No resume checkpoint found. Training from scratch.")

    # Completion check based on step budgets
    if training_complete or (
        stage_a_steps_done >= stage_a_step_budget and stage_b_steps_done >= stage_b_step_budget
    ):
        print("[INFO] Training already completed. Running final test evaluation only.")
        best_ckpt = load_resume(BEST_CHECKPOINT, device)
        if best_ckpt is not None:
            model.load_state_dict(best_ckpt["model_state"])
        test_acc = validate(model, test_loader, device, class_names, split_name="test(best)")
        print(f"Final test accuracy: {test_acc:.4f}")
        return

    def save_epoch_ckpt(epoch, stage, optimizer, scheduler, scaler):
        nonlocal training_complete
        training_complete = (
            stage_a_steps_done >= stage_a_step_budget and stage_b_steps_done >= stage_b_step_budget
        )
        ckpt_data = {
            "epoch": epoch,
            "stage": stage,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_acc": best_acc,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "class_to_idx": class_to_idx,
            "experiment": args.experiment_name,
            "backbone_init": args.backbone_init,
            "label_fraction": args.label_fraction,
            "global_step": global_step,
            "stage_a_steps_done": stage_a_steps_done,
            "stage_b_steps_done": stage_b_steps_done,
            "total_step_budget": total_step_budget,
            "stage_a_step_budget": stage_a_step_budget,
            "stage_b_step_budget": stage_b_step_budget,
            "stage_a_precision": stage_a_precision,
            "stage_b_precision": stage_b_precision,
            "training_complete": training_complete,
            "step_budget_mode": args.step_budget_mode,
            "max_steps": args.max_steps,
            "step_budget_reference_fraction": args.step_budget_reference_fraction,
        }
        save_checkpoint(ckpt_data, RESUME_CHECKPOINT)
        return ckpt_data

    # -----------------------
    # Stage A
    # -----------------------
    set_backbone_trainable(model, trainable=False)
    optimizer = build_optimizer(model, stage="A", backbone_init=args.backbone_init)
    scheduler = None
    scaler = build_grad_scaler(stage_a_precision, num_train_batches=len(train_loader))
    if scaler is not None:
        print(f"[INFO] Stage A GradScaler initialized (scale={get_scaler_scale_str(scaler)})")

    skip_stage_a = stage_a_steps_done >= stage_a_step_budget
    if resume is not None and not skip_stage_a and start_stage == "A":
        if "optimizer_state" in resume and resume["optimizer_state"] is not None:
            try:
                optimizer.load_state_dict(resume["optimizer_state"])
                print("[INFO] Stage A optimizer state restored.")
            except Exception as exc:
                print(f"[WARN] Could not restore Stage A optimizer state: {exc}")
        if scaler is not None and resume.get("scaler_state") is not None:
            try:
                scaler.load_state_dict(resume["scaler_state"])
                print(f"[INFO] Stage A scaler state restored (scale={get_scaler_scale_str(scaler)})")
            except Exception as exc:
                print(f"[WARN] Could not restore Stage A scaler state: {exc}")

    stage_a_epoch_est = max(1, math.ceil(stage_a_step_budget / max(1, len(train_loader)))) if stage_a_step_budget > 0 else 0
    stage_a_epoch_idx = max(0, stage_a_steps_done // max(1, len(train_loader)))
    stage_a_recoveries = 0

    while not skip_stage_a and stage_a_steps_done < stage_a_step_budget:
        stage_a_epoch_idx += 1
        logical_epoch += 1
        epoch_start_steps = stage_a_steps_done

        out = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            precision_mode=stage_a_precision,
            scaler=scaler,
            stage_name="StageA",
            epoch_idx=stage_a_epoch_idx,
            epoch_total_est=stage_a_epoch_est,
            steps_done=stage_a_steps_done,
            step_budget=stage_a_step_budget,
            scheduler=None,
        )
        stage_a_steps_done = out["steps_done"]
        global_step = stage_a_steps_done + stage_b_steps_done
        no_progress = stage_a_steps_done == epoch_start_steps

        if no_progress:
            print(
                f"[WARN] Stage A made no optimizer-step progress in epoch {stage_a_epoch_idx} "
                f"(likely due repeated non-finite gradients)."
            )

        if out["nonfinite_param_event"] or no_progress:
            stage_a_recoveries += 1
            if stage_a_recoveries > args.max_recoveries_per_stage:
                raise RuntimeError("Stage A exceeded max_recoveries_per_stage after non-finite parameter events.")

            restored_ckpt, restored_source, _ = recover_from_last_finite_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                resume_path=RESUME_CHECKPOINT,
                best_path=BEST_CHECKPOINT,
                stage="A",
            )
            reduce_optimizer_lrs(optimizer, args.recovery_lr_factor)
            safer_mode = downgrade_precision_mode(stage_a_precision, device)
            if safer_mode != stage_a_precision:
                print(f"[INFO] Stage A precision downgraded: {stage_a_precision} -> {safer_mode}")
                stage_a_precision = safer_mode
                scaler = build_grad_scaler(stage_a_precision, num_train_batches=len(train_loader))
            elif scaler is not None:
                scaler = build_grad_scaler(stage_a_precision, num_train_batches=len(train_loader))

            stage_a_steps_done = int(restored_ckpt.get("stage_a_steps_done", epoch_start_steps))
            stage_b_steps_done = int(restored_ckpt.get("stage_b_steps_done", stage_b_steps_done))
            global_step = int(restored_ckpt.get("global_step", stage_a_steps_done + stage_b_steps_done))
            logical_epoch = int(restored_ckpt.get("epoch", logical_epoch - 1))
            print(
                f"[INFO] Stage A recovered from {restored_source}; "
                f"steps reset to A={stage_a_steps_done}, B={stage_b_steps_done}."
            )
            continue

        val_acc = validate(model, val_loader, device, class_names)
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        print_resource_usage(f"A-E{logical_epoch}")
        print(f"[INFO] Stage A scaler={get_scaler_scale_str(scaler)} | grad_nf_batches={out['nonfinite_grad_batches']}")

        metrics_logger.log(
            logical_epoch,
            stage="A",
            train_loss=out["train_loss"],
            train_acc=out["train_acc"],
            val_acc=val_acc,
            best_acc=best_acc,
            lr=optimizer.param_groups[0]["lr"],
            stage_a_steps_done=stage_a_steps_done,
            stage_b_steps_done=stage_b_steps_done,
            precision=stage_a_precision,
            grad_nonfinite_batches=out["nonfinite_grad_batches"],
        )

        ckpt_data = save_epoch_ckpt(
            epoch=logical_epoch,
            stage="A",
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if is_best:
            save_checkpoint(ckpt_data, BEST_CHECKPOINT)

    # -----------------------
    # Stage B setup
    # -----------------------
    del train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_loader, val_loader, test_loader, _ = build_data_loaders(
        DATA_ROOT, INPUT_SIZE, BATCH_SIZE, NUM_WORKERS, class_to_idx,
        label_fraction=args.label_fraction, eval_batch_size=EVAL_BATCH_SIZE, seed=args.seed,
    )
    print("[INFO] Data loaders rebuilt for Stage B")

    set_backbone_trainable(model, trainable=True)
    optimizer = build_optimizer(model, stage="B", backbone_init=args.backbone_init)

    scheduler = None
    if stage_b_step_budget > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, stage_b_step_budget)
        )

    scaler = build_grad_scaler(stage_b_precision, num_train_batches=len(train_loader))
    if scaler is not None:
        print(f"[INFO] Stage B GradScaler initialized (scale={get_scaler_scale_str(scaler)})")
    else:
        print("[INFO] Stage B running without GradScaler.")

    if resume is not None and stage_b_steps_done > 0:
        if "optimizer_state" in resume and resume["optimizer_state"] is not None:
            try:
                optimizer.load_state_dict(resume["optimizer_state"])
                print("[INFO] Stage B optimizer state restored.")
            except Exception as exc:
                print(f"[WARN] Could not restore Stage B optimizer state: {exc}")
        if scheduler is not None and "scheduler_state" in resume and resume["scheduler_state"] is not None:
            try:
                scheduler.load_state_dict(resume["scheduler_state"])
                print("[INFO] Stage B scheduler state restored.")
            except Exception as exc:
                print(f"[WARN] Could not restore Stage B scheduler state: {exc}")
        if scaler is not None and resume.get("scaler_state") is not None:
            try:
                scaler.load_state_dict(resume["scaler_state"])
                print(f"[INFO] Stage B scaler state restored (scale={get_scaler_scale_str(scaler)})")
            except Exception as exc:
                print(f"[WARN] Could not restore Stage B scaler state: {exc}")

    stage_b_epoch_est = max(1, math.ceil(stage_b_step_budget / max(1, len(train_loader)))) if stage_b_step_budget > 0 else 0
    stage_b_epoch_idx = max(0, stage_b_steps_done // max(1, len(train_loader)))
    stage_b_recoveries = 0

    while stage_b_steps_done < stage_b_step_budget:
        stage_b_epoch_idx += 1
        logical_epoch += 1
        epoch_start_steps = stage_b_steps_done

        out = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            precision_mode=stage_b_precision,
            scaler=scaler,
            stage_name="StageB",
            epoch_idx=stage_b_epoch_idx,
            epoch_total_est=stage_b_epoch_est,
            steps_done=stage_b_steps_done,
            step_budget=stage_b_step_budget,
            scheduler=scheduler,
        )
        stage_b_steps_done = out["steps_done"]
        global_step = stage_a_steps_done + stage_b_steps_done
        no_progress = stage_b_steps_done == epoch_start_steps

        if no_progress:
            print(
                f"[WARN] Stage B made no optimizer-step progress in epoch {stage_b_epoch_idx} "
                f"(likely due repeated non-finite gradients)."
            )

        if out["nonfinite_param_event"] or no_progress:
            stage_b_recoveries += 1
            if stage_b_recoveries > args.max_recoveries_per_stage:
                raise RuntimeError("Stage B exceeded max_recoveries_per_stage after non-finite parameter events.")

            restored_ckpt, restored_source, _ = recover_from_last_finite_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                resume_path=RESUME_CHECKPOINT,
                best_path=BEST_CHECKPOINT,
                stage="B",
            )
            reduce_optimizer_lrs(optimizer, args.recovery_lr_factor)
            safer_mode = downgrade_precision_mode(stage_b_precision, device)
            if safer_mode != stage_b_precision:
                print(f"[INFO] Stage B precision downgraded: {stage_b_precision} -> {safer_mode}")
                stage_b_precision = safer_mode
            scaler = build_grad_scaler(stage_b_precision, num_train_batches=len(train_loader))

            stage_a_steps_done = int(restored_ckpt.get("stage_a_steps_done", stage_a_steps_done))
            stage_b_steps_done = int(restored_ckpt.get("stage_b_steps_done", epoch_start_steps))
            global_step = int(restored_ckpt.get("global_step", stage_a_steps_done + stage_b_steps_done))
            logical_epoch = int(restored_ckpt.get("epoch", logical_epoch - 1))
            print(
                f"[INFO] Stage B recovered from {restored_source}; "
                f"steps reset to A={stage_a_steps_done}, B={stage_b_steps_done}; "
                f"new precision={stage_b_precision}."
            )
            continue

        val_acc = validate(model, val_loader, device, class_names)
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        print_resource_usage(f"B-E{logical_epoch}")
        print(f"[INFO] Stage B scaler={get_scaler_scale_str(scaler)} | grad_nf_batches={out['nonfinite_grad_batches']}")

        metrics_logger.log(
            logical_epoch,
            stage="B",
            train_loss=out["train_loss"],
            train_acc=out["train_acc"],
            val_acc=val_acc,
            best_acc=best_acc,
            lr=optimizer.param_groups[0]["lr"],
            stage_a_steps_done=stage_a_steps_done,
            stage_b_steps_done=stage_b_steps_done,
            precision=stage_b_precision,
            grad_nonfinite_batches=out["nonfinite_grad_batches"],
        )

        ckpt_data = save_epoch_ckpt(
            epoch=logical_epoch,
            stage="B",
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if is_best:
            save_checkpoint(ckpt_data, BEST_CHECKPOINT)

    # ensure final resume checkpoint marks completion
    final_ckpt = save_epoch_ckpt(
        epoch=logical_epoch,
        stage="B" if stage_b_step_budget > 0 else "A",
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    final_ckpt["training_complete"] = True
    save_checkpoint(final_ckpt, RESUME_CHECKPOINT)

    best_ckpt = load_resume(BEST_CHECKPOINT, device)
    if best_ckpt is None:
        print("[WARN] Best checkpoint not found. Falling back to last weights.")
        test_acc = validate(model, test_loader, device, class_names, split_name="test(last)")
    else:
        model.load_state_dict(best_ckpt["model_state"])
        test_acc = validate(model, test_loader, device, class_names, split_name="test(best)")

    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Results saved to: {CHECKPOINT_DIR}")


@torch.no_grad()
def validate(model, loader, device, class_names=None, split_name="val"):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        _, preds = logits.max(1)
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"Total samples in {split_name}: {total_samples}")
    print(f"{split_name.capitalize()} accuracy: {acc:.4f}")

    if class_names is not None and total_samples > 0:
        unique_labels = np.unique(all_labels)
        unique_preds = np.unique(all_preds)
        print(
            f"[DEBUG] Unique labels in {split_name}: {len(unique_labels)}, "
            f"range: [{min(all_labels)}, {max(all_labels)}]"
        )
        print(
            f"[DEBUG] Unique predictions: {len(unique_preds)}, "
            f"range: [{min(all_preds)}, {max(all_preds)}]"
        )

    return acc


if __name__ == "__main__":
    train()
