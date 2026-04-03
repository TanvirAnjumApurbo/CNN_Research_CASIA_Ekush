import os
import gc
import sys
import copy
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from lightly.data import LightlyDataset, DINOCollateFunction
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.loss import NegativeCosineSimilarity

# =========================
# CONFIGURATION (defaults, overridable via CLI)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_CHECKPOINT_PATH = str(PROJECT_ROOT / "chkp" / "checkpoint_2.pth")

BACKBONE = "resnet50"
BATCH_SIZE = 192  # Peak VRAM was 6.2GB/8GB at batch=192 -- fits well
INPUT_SIZE = 128
EPOCHS = 100      # 100 epochs sufficient for large datasets (817K+)
LR = 0.001
NUM_WORKERS = 0   # 0 is fastest on Windows 16GB RAM -- avoids worker process RAM pressure
GRAD_ACCUM_STEPS = 1  # batch 192 is already a good effective batch


def parse_args():
    parser = argparse.ArgumentParser(description="BYOL Self-Supervised Pretraining")
    parser.add_argument(
        "--data_root", type=str, default=str(DEFAULT_DATA_ROOT),
        help="Root directory containing a 'train' folder (or image subfolders directly)"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH,
        help="Path to save/resume BYOL checkpoint"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)"
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50"],
        help="Backbone architecture"
    )
    parser.add_argument(
        "--subsample", type=float, default=0.25,
        help="Fraction of dataset to use per epoch (0.25 = 25%%). Random subset reshuffled each epoch. "
             "Set to 1.0 to use full dataset. For large datasets (800K+), 0.25 is recommended."
    )
    return parser.parse_args()

# Force standard CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OPTIMIZATION: Enable cuDNN benchmarking for speed
torch.backends.cudnn.benchmark = True


class BYOL(nn.Module):
    def __init__(self, backbone, feature_dim):
        super().__init__()
        self.backbone = backbone

        # ResNet50 outputs 2048 dim features (ResNet18 was 512)
        self.projection_head = BYOLProjectionHead(feature_dim, 4096, 256)
        self.prediction_head = BYOLPredictionHead(256, 4096, 256)

        self.backbone_momentum = copy.deepcopy(backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        return z.detach()


def load_checkpoint(model, optimizer, scaler, path):
    if not os.path.exists(path):
        return 0

    print(f"🔄 Loading checkpoint from {path}")
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Load scaler state if it exists (important for AMP)
        if "scaler_state" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])

        epoch = checkpoint.get("epoch", 0)
        return epoch + 1
    except Exception as e:
        print(f"⚠ Checkpoint corrupted or incompatible: {e}")
        return 0


def train(args=None):
    if args is None:
        args = parse_args()

    checkpoint_path = args.checkpoint_path
    data_root = Path(args.data_root)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    input_size = args.input_size
    num_workers = args.num_workers
    grad_accum_steps = args.grad_accum_steps
    backbone_name = args.backbone

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if device.type == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    subsample_frac = args.subsample

    print(f"Config: backbone={backbone_name}, batch={batch_size}, accum={grad_accum_steps}, "
          f"effective_batch={batch_size * grad_accum_steps}, epochs={epochs}, lr={lr}, "
          f"subsample={subsample_frac:.0%}")

    # 1. Prepare Data
    collate_fn = DINOCollateFunction(
        input_size, gaussian_blur=(0.0, 0.1, 2.0), solarization_prob=0.0, hf_prob=0.5
    )

    # Auto-detect train folder
    train_dir = str(data_root)
    for root, dirs, _ in os.walk(data_root):
        if "train" in dirs:
            train_dir = os.path.join(root, "train")
            break

    print(f"Loading data from: {train_dir}")
    full_dataset = LightlyDataset(input_dir=train_dir)
    full_size = len(full_dataset)
    print(f"Full dataset size: {full_size} images")

    # Subsample to speed up training using a custom sampler.
    # The DataLoader is created ONCE -- no per-epoch recreation, no RAM spikes.
    use_subsample = subsample_frac < 1.0
    subset_size = int(full_size * subsample_frac) if use_subsample else full_size
    if use_subsample:
        print(f"Using {subsample_frac:.0%} subsample per epoch: {subset_size} images "
              f"(random subset reshuffled each epoch via sampler, no DataLoader recreation)")

    class EpochSubsetSampler(torch.utils.data.Sampler):
        """Draws a random subset of indices each epoch without rebuilding the DataLoader."""
        def __init__(self, n_total, n_subset):
            self.n_total = n_total
            self.n_subset = n_subset
            self.epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            rng = np.random.RandomState(self.epoch)
            idx = rng.choice(self.n_total, size=self.n_subset, replace=False)
            return iter(idx.tolist())

        def __len__(self):
            return self.n_subset

    import platform
    is_windows = platform.system() == "Windows"
    # On Windows with 16GB RAM: num_workers=0 is fastest.
    # Worker processes each consume ~500MB+ RAM; with 80%+ RAM usage at start
    # they push Windows into memory compression, causing 20-80x slowdowns.
    effective_workers = 0 if is_windows else num_workers

    sampler = EpochSubsetSampler(full_size, subset_size)
    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=sampler,   # custom sampler controls subset + order; shuffle must be False
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=effective_workers,
        pin_memory=True,
        prefetch_factor=None,  # must be None when num_workers=0
    )
    print(f"DataLoader: {len(dataloader)} batches/epoch, workers={effective_workers}")

    # 2. Setup Model
    if backbone_name == "resnet18":
        resnet = torchvision.models.resnet18(weights=None)
        feature_dim = 512
    else:
        resnet = torchvision.models.resnet50(weights=None)
        feature_dim = 2048

    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = BYOL(backbone, feature_dim=feature_dim)

    # Channels Last memory format (better for conv layers on NVIDIA)
    model = model.to(device, memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = NegativeCosineSimilarity()

    # AMP Scaler for Mixed Precision
    scaler = torch.amp.GradScaler("cuda")

    # 3. Load Checkpoint
    start_epoch = load_checkpoint(model, optimizer, scaler, checkpoint_path)
    if start_epoch >= epochs:
        print("Training already completed.")
        return

    # Metrics logger
    from utils import MetricsLogger, print_resource_usage
    log_path = Path(checkpoint_path).parent / (Path(checkpoint_path).stem + "_metrics.json")
    metrics_logger = MetricsLogger(log_path)

    # 4. Training Loop
    print(f"Starting Training from Epoch {start_epoch}...")
    print_resource_usage("INIT")
    model.train()

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)  # new random subset indices, no DataLoader recreation

        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            views, _, _ = batch

            x0 = views[0].to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
            x1 = views[1].to(
                device, non_blocking=True, memory_format=torch.channels_last
            )

            update_momentum(model.backbone, model.backbone_momentum, m=0.99)
            update_momentum(
                model.projection_head, model.projection_head_momentum, m=0.99
            )

            with torch.amp.autocast("cuda"):
                p0 = model(x0)
                z0 = model.forward_momentum(x0)
                p1 = model(x1)
                z1 = model.forward_momentum(x1)
                loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * grad_accum_steps
            pbar.set_postfix(loss=loss.item() * grad_accum_steps)

        avg_loss = total_loss / len(dataloader)
        # Free accumulated memory BEFORE it snowballs into next epoch.
        # Without this, CUDA caching allocator + Windows shared GPU memory
        # eat system RAM (~10%/epoch), causing paging by epoch 2.
        del pbar
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        res = print_resource_usage(f"E{epoch+1}")
        metrics_logger.log(epoch, train_loss=avg_loss, **{k: v for k, v in res.items() if isinstance(v, (int, float))})

        try:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "backbone": backbone_name,
                    "feature_dim": feature_dim,
                },
                checkpoint_path,
            )
            print("Checkpoint saved.")
        except Exception as e:
            print(f"Save failed: {e}")

        # On Windows with 16GB RAM, Python's pymalloc never returns memory
        # to the OS after torch.save(). After one epoch RAM grows 5-10%,
        # causing Windows memory compression and 20-40x slowdowns.
        # Fix: exit after each epoch. Use the provided PowerShell loop to
        # auto-resume from checkpoint:
        #   while ($true) { python ssl_train.py <args>; if ($LASTEXITCODE -ne 0) { break } }
        if epoch + 1 < epochs:
            print(f"\n[EPOCH DONE] {epoch+1}/{epochs} complete. "
                  f"Exiting for clean restart (avoids Windows RAM bloat).")
            sys.exit(0)

    print(f"\n[TRAINING COMPLETE] All {epochs} epochs finished.")
    sys.exit(1)  # exit 1 = done, stops the PowerShell loop


if __name__ == "__main__":
    train()
