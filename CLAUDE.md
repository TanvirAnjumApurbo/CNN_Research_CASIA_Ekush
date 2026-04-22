# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Platform

This is a Windows development environment (Windows 11, PowerShell). Always account for Windows-specific limitations:
- `torch.compile` and Triton are NOT supported on Windows
- Use single slashes for PowerShell flags (e.g., `taskkill /F /PID`), not double slashes
- `os.execl` is unreliable on Windows; avoid it
- `persistent_workers=True` in DataLoader can cause page file exhaustion on low-RAM systems
- Always use `num_workers=0` when RAM caching is enabled
- Python's pymalloc never returns memory to the OS after large allocations (e.g., `torch.save`); for long-running training, use process-per-epoch restart via `sys.exit(0)` + PowerShell loop

## Hardware Constraints

System has 16GB RAM, RTX 5060 Ti (8GB VRAM). When writing ML code:
- Cache datasets as uint8, never float32 (float32 caching can use 25GB+ RAM)
- Set `num_workers=0` when using in-memory caching; cap at 2 on Windows otherwise
- Avoid duplicate DataLoader rebuilds that cause double caching
- Always consider memory footprint before suggesting caching strategies
- batch_size=128 for fine-tuning, 192 for SSL pretraining (fits 8GB VRAM with AMP)

## Debugging Approach

When fixing errors, always check the actual codebase first before giving generic advice. When a fix fails, do NOT just tweak the same approach — consider whether the approach itself is wrong for this platform/hardware. After applying a fix, verify there are no inconsistencies with the rest of the codebase (e.g., mismatched checkpoint keys, missing imports, mismatched normalization stats).

## Project Overview

Cross-script transfer learning study: can BYOL self-supervised pretraining on Chinese handwriting (CASIA) transfer to Bangla handwriting recognition (Ekush), reducing labeled data requirements?

**Datasets** (not in repo, stored on local drives):
- CASIA: `C:\CASIA` — 3,891 Chinese character classes, ~1.17M images (NVMe SSD)
- Ekush: `C:\Ekush\dataset\dataset` — 237 Bangla character classes, ~592K images (NVMe SSD)

Both have `train/`, `val/`, `test/` subdirectories with numeric folder names.

## Architecture

The model pipeline: `Input Image → ResNet50 Backbone (2048ch, 4×4) → TransformerHead (4 layers, d=512, CLS token) → Embedding FC (Linear + BatchNorm1d) → ReLU → Dropout(0.1) → Linear(512, 237) → logits`

All defined in `src/model.py`. Key classes: `ResNet50Backbone`, `TransformerHead`, `HybridModel`.

**AMP stability rule** — `embedding_fc`'s `BatchNorm1d` MUST run in fp32 even under mixed precision (running statistics drift in fp16, causing NaN after ~8 epochs). Uses `torch.amp.autocast(device_type="cuda", enabled=False)` + `.float()` cast.

## Training Pipeline

All scripts run from `src/`. The pipeline has 3 phases:

**Phase 1 — Pretraining** (produces backbone weights):
- `ssl_train.py` — BYOL on CASIA or Ekush. Uses `EpochSubsetSampler` for subsampling. On Windows, run with PowerShell loop: `while ($true) { python ssl_train.py <args>; if ($LASTEXITCODE -ne 0) { break } }`
- `supervised_pretrain.py` — Supervised on CASIA (3,891 classes)

**Phase 2 — Fine-tuning** on Ekush:
- `train_hybrid_small.py` — Two-stage training. Stage A (6 epochs): backbone frozen. Stage B (94 epochs): full fine-tune with differential LR (backbone 1e-5, head 1e-4). For `random` backbone_init, Stage A is skipped (freezing random features is counterproductive) and backbone LR is raised to 1e-4.

**Phase 3 — Evaluation**:
- `evaluate_model.py` — Per-experiment metrics, confusion matrix, ROC, PR curves
- `visualize_embeddings.py` — t-SNE scatter plots
- `plot_results.py` — Cross-experiment comparison figures

**Orchestration**: `run_experiments.py --phase finetune` runs all 15 experiments (5 inits × 3 fractions: 1%, 10%, 100%). Skip logic checks `resume_checkpoint.pth` epoch count.

## Checkpoint System

Every training script saves `resume_checkpoint.pth` (latest) and `best_checkpoint.pth` (best val_acc). Checkpoints include: `epoch`, `stage`, `model_state`, `optimizer_state`, `scaler_state`, `scheduler_state`, `best_acc`, `class_to_idx`.

Checkpoint layout:
```
chkp/ssl_pretrain/byol_casia.pth, byol_ekush.pth
chkp/supervised_pretrain/backbone_weights.pth
chkp/experiments/<exp_name>/best_checkpoint.pth, resume_checkpoint.pth
```

SSL backbone loading uses `extract_backbone_state()` which tries multiple key prefixes (`backbone.`, `module.`, etc.) to handle different checkpoint formats.

## Data Pipeline

`dataset.py` provides `AlbumentationsImageFolder` wrapping torchvision's `ImageFolder` with Albumentations transforms. All images are converted to grayscale (`ToGray(p=1.0)`) then repeated to 3 channels. Normalization: mean=0.5, std=0.5.

`FixedClassImageFolder` (in both `train_hybrid_small.py` and `supervised_pretrain.py`) enforces numeric sorting of class folders so class-to-index mapping is consistent across splits.

## Key Patterns

- **Validation protocol**: `model(imgs)` returns logits directly. For embedding extraction (t-SNE), use `model(imgs, return_embedding=True)`.
- **Gradient clipping**: `scaler.unscale_(optimizer)` then `clip_grad_norm_(model.parameters(), max_norm=1.0)` before `scaler.step()`.
- **Journal figures**: `plot_config.py` defines the color palette, `apply_journal_style()`, and `save_figure()` (outputs both PNG 300dpi + PDF). No titles on figures — captions go in LaTeX.
- **Experiment naming**: `{backbone_init}_frac{fraction:.2f}` (e.g., `ssl_casia_frac0.10`)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install lightly albumentations

# Run all fine-tuning experiments (from src/)
python run_experiments.py --phase finetune

# Run a single experiment
python train_hybrid_small.py --data_root "C:\Ekush\dataset\dataset" --backbone_init ssl --ssl_checkpoint ../chkp/ssl_pretrain/byol_casia.pth --label_fraction 0.10 --experiment_name ssl_casia_frac0.10 --batch_size 128

# Evaluate a trained model
python evaluate_model.py --model_path ../chkp/experiments/ssl_casia_frac1.00/best_checkpoint.pth --data_root "C:\Ekush\dataset\dataset"

# Generate cross-experiment plots
python plot_results.py --results_dir ../chkp/experiments --output_dir ../reports
```
