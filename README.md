# Cross-Script Transfer Learning for Handwritten Bangla Character Recognition

**Research Question:** Can self-supervised representations learned from Chinese handwriting (CASIA, 3,891 classes) transfer effectively to Bangla handwriting recognition (Ekush, 237 classes), and does cross-script pretraining reduce labeled data requirements compared to standard ImageNet pretraining?

---

## Methodology Overview

```
+===========================================================================+
|                        CROSS-SCRIPT TRANSFER LEARNING                     |
|                   Handwritten Bangla Character Recognition                 |
+===========================================================================+

 PHASE 1: PRETRAINING (5 backbone initialization strategies)
 ============================================================

 Strategy A: Random Initialization
 +------------------+
 | ResNet50 Backbone|  (no pretraining, trained from scratch)
 | (random weights) |
 +------------------+

 Strategy B: ImageNet Pretrained
 +------------------+
 | ResNet50 Backbone|  (standard transfer learning baseline)
 | (ImageNet init)  |
 +------------------+

 Strategy C: BYOL Self-Supervised on CASIA (Cross-Script SSL)
 +---------------------------+     +------------------+
 | CASIA Chinese Handwriting |     |  BYOL Framework  |
 |   3,891 classes           | --> |  Online Network  |
 |   1.17M images            |     |  + Momentum Net  |
 +---------------------------+     +--------+---------+
                                            |
                                   +--------v---------+
                                   | ResNet50 Backbone |
                                   | (learned features)|
                                   +------------------+

 Strategy D: BYOL Self-Supervised on Ekush (In-Domain SSL)
 +---------------------------+     +------------------+
 | Ekush Bangla Handwriting  |     |  BYOL Framework  |
 |   237 classes             | --> |  Online Network  |
 |   592K images             |     |  + Momentum Net  |
 +---------------------------+     +--------+---------+
                                            |
                                   +--------v---------+
                                   | ResNet50 Backbone |
                                   | (learned features)|
                                   +------------------+

 Strategy E: Supervised Pretraining on CASIA
 +---------------------------+     +-------------------+
 | CASIA Chinese Handwriting |     | ResNet50 + GAP    |
 |   3,891 classes           | --> | + Linear(2048,    |
 |   1.17M images            |     |          3891)    |
 +---------------------------+     +--------+----------+
                                            |
                                   +--------v---------+
                                   | ResNet50 Backbone |
                                   | (supervised feat.)|
                                   +------------------+


 PHASE 2: FINE-TUNING ON EKUSH (Target Dataset)
 ============================================================

 Each pretrained backbone feeds into the same downstream model:

 +------------------+     +-------------------+     +------------------+
 | 128x128 Bangla   |     | ResNet50 Backbone |     | Transformer Head |
 | Character Image  | --> | (from Phase 1)    | --> | 4 encoder layers |
 | (grayscale->3ch) |     | Output: 2048ch    |     | d_model=512      |
 +------------------+     | 4x4 feature map   |     | nhead=8          |
                          +-------------------+     | + CLS token      |
                                                    +--------+---------+
                                                             |
                                                    +--------v---------+
                                                    | Embedding FC     |
                                                    | Linear(512,512)  |
                                                    | + BatchNorm      |
                                                    +--------+---------+
                                                             |
                                                    +--------v---------+
                                                    | ArcFace Head     |
                                                    | s=30.0, m=0.3   |
                                                    | 237 classes      |
                                                    +------------------+

 Two-Stage Training:
   Stage A (6 epochs):   Backbone FROZEN, train Transformer + ArcFace only
   Stage B (194 epochs): Full model fine-tuning with differential LR
                         - Backbone LR:    1e-5
                         - Transformer LR: 1e-4
                         - Head LR:        1e-4


 PHASE 3: LABEL EFFICIENCY EXPERIMENTS
 ============================================================

 For EACH of the 5 pretraining strategies, fine-tune with:

   +-------+------+------+------+------+-------+
   |  1%   |  5%  | 10%  | 25%  | 50%  | 100%  |  <-- % of labeled Ekush data
   +-------+------+------+------+------+-------+

   = 5 strategies x 6 fractions = 30 experiments total

   Key output: Accuracy vs. Label Fraction curves
   --> "Cross-script SSL pretraining with X% labels matches
        ImageNet pretraining with 100% labels"


 PHASE 4: EVALUATION & ANALYSIS
 ============================================================

 For each experiment:
   - Top-1 and Top-5 accuracy
   - Per-class precision, recall, F1
   - Confusion matrices
   - ROC and Precision-Recall curves
   - t-SNE embedding visualization

 Cross-experiment comparisons:
   - Label efficiency curves (accuracy vs label fraction)
   - Pretraining strategy comparison bar charts
   - Training dynamics overlay (loss, accuracy curves)
   - Label savings quantification
```

---

## Project Structure

```
test_research/
|-- data/
|   |-- CASIA/                      # Chinese handwriting (source dataset)
|   |   |-- train/                   #   3,891 classes, ~1.17M images
|   |   |-- val/
|   |   +-- test/
|   +-- Ekush/dataset/dataset/       # Bangla handwriting (target dataset)
|       |-- train/                   #   237 classes, ~592K images
|       |-- val/
|       +-- test/
|
|-- src/
|   |-- model.py                    # ResNet50Backbone, TransformerHead, ArcFace, HybridModel
|   |-- dataset.py                  # AlbumentationsImageFolder, get_transforms
|   |-- utils.py                    # MetricsLogger, checkpointing, resource monitoring
|   |-- plot_config.py              # Journal color palette, matplotlib rcParams, save_figure()
|   |-- ssl_train.py                # BYOL self-supervised pretraining
|   |-- supervised_pretrain.py      # Supervised pretraining on CASIA (3,891 classes)
|   |-- train_hybrid_small.py       # Two-stage fine-tuning with label fraction support
|   |-- evaluate_model.py           # Full evaluation: metrics, confusion matrix, ROC, PR curves
|   |-- visualize_embeddings.py     # t-SNE embedding visualization
|   |-- plot_results.py             # Cross-experiment comparison figures
|   +-- run_experiments.py          # Orchestration: runs full experimental matrix
|
|-- chkp/                           # Checkpoints (auto-created)
|   |-- ssl_pretrain/               #   BYOL checkpoints (byol_casia.pth, byol_ekush.pth)
|   |-- supervised_pretrain/        #   Supervised CASIA checkpoint + backbone_weights.pth
|   +-- experiments/                #   Per-experiment dirs (e.g., ssl_casia_frac0.10/)
|       +-- <experiment_name>/
|           |-- resume_checkpoint.pth
|           +-- best_checkpoint.pth
|
|-- reports/                        # All generated figures and results (auto-created)
|   |-- embeddings/                 #   t-SNE plots and raw .npz files
|   +-- *.png / *.pdf               #   All figures in both formats
|
|-- requirements.txt
+-- README.md
```

---

## File Run Order

### Quick Reference (Run from `src/` directory)

| Step | Script | Purpose | Approx. Time |
|------|--------|---------|-------------|
| 1 | `ssl_train.py` | BYOL pretraining on CASIA | ~3-4 days |
| 2 | `ssl_train.py` | BYOL pretraining on Ekush | ~1.5-2 days |
| 3 | `supervised_pretrain.py` | Supervised pretraining on CASIA | ~1-2 days |
| 4 | `train_hybrid_small.py` | Fine-tune on Ekush (x30 configs) | ~10h each |
| 5 | `evaluate_model.py` | Generate per-experiment evaluation | ~10 min each |
| 6 | `visualize_embeddings.py` | t-SNE for each experiment | ~5 min each |
| 7 | `plot_results.py` | Cross-experiment comparison plots | ~2 min |

### Detailed Commands

**Step 1: BYOL on CASIA (cross-script SSL)**
```bash
python ssl_train.py \
  --data_root ../data/CASIA \
  --checkpoint_path ../chkp/ssl_pretrain/byol_casia.pth \
  --batch_size 64 --grad_accum_steps 2 --epochs 200 --backbone resnet50
```

**Step 2: BYOL on Ekush (in-domain SSL)**
```bash
python ssl_train.py \
  --data_root ../data/Ekush/dataset/dataset \
  --checkpoint_path ../chkp/ssl_pretrain/byol_ekush.pth \
  --batch_size 64 --grad_accum_steps 2 --epochs 200 --backbone resnet50
```

**Step 3: Supervised pretraining on CASIA**
```bash
python supervised_pretrain.py \
  --data_root ../data/CASIA \
  --checkpoint_dir ../chkp/supervised_pretrain \
  --epochs 50 --batch_size 64 --lr 0.001
```

**Step 4: Fine-tuning experiments (example for one config)**
```bash
# Random init, 100% labels
python train_hybrid_small.py \
  --data_root ../data/Ekush/dataset/dataset \
  --backbone_init random --label_fraction 1.0 \
  --experiment_name random_frac1.00

# SSL CASIA init, 10% labels
python train_hybrid_small.py \
  --data_root ../data/Ekush/dataset/dataset \
  --backbone_init ssl \
  --ssl_checkpoint ../chkp/ssl_pretrain/byol_casia.pth \
  --label_fraction 0.10 \
  --experiment_name ssl_casia_frac0.10

# ImageNet init, 100% labels
python train_hybrid_small.py \
  --data_root ../data/Ekush/dataset/dataset \
  --backbone_init imagenet --label_fraction 1.0 \
  --experiment_name imagenet_frac1.00

# Supervised CASIA init, 25% labels
python train_hybrid_small.py \
  --data_root ../data/Ekush/dataset/dataset \
  --backbone_init ssl \
  --ssl_checkpoint ../chkp/supervised_pretrain/backbone_weights.pth \
  --label_fraction 0.25 \
  --experiment_name supervised_casia_frac0.25
```

**Step 5: Evaluation**
```bash
python evaluate_model.py \
  --model_path ../chkp/experiments/ssl_casia_frac1.00/best_checkpoint.pth \
  --data_root ../data/Ekush/dataset/dataset \
  --output_dir ../reports/ssl_casia_frac1.00
```

**Step 6: t-SNE Visualization**
```bash
python visualize_embeddings.py \
  --model_path ../chkp/experiments/ssl_casia_frac1.00/best_checkpoint.pth \
  --data_root ../data/Ekush/dataset/dataset \
  --split test --max_samples 5000 \
  --output_dir ../reports/embeddings/ssl_casia_frac1.00 \
  --experiment_name ssl_casia_frac1.00
```

**Step 7: Cross-experiment comparison plots**
```bash
python plot_results.py --results_dir ../chkp/experiments --output_dir ../reports
```

### Automated (Full Pipeline)

```bash
# Run everything sequentially
python run_experiments.py --phase all

# Or run individual phases
python run_experiments.py --phase pretrain
python run_experiments.py --phase finetune
python run_experiments.py --phase visualize
python run_experiments.py --phase collect
```

---

## BYOL (Bootstrap Your Own Latent) Architecture

```
                     BYOL Self-Supervised Learning
                     =============================

 Input Image
      |
      v
 +----+-----+                    +----------+
 | Augment  |                    | Augment  |
 | View 1   |                    | View 2   |
 +----+-----+                    +----+-----+
      |                               |
      v                               v
 +----+----------+              +-----+---------+
 | ONLINE        |              | MOMENTUM      |
 | Network       |              | Network       |
 |               |              | (EMA of       |
 | ResNet50      |              |  Online)      |
 | Backbone      |              |               |
 |     |         |              | ResNet50      |
 |     v         |              | Backbone      |
 | Projection    |              |     |         |
 | Head          |              |     v         |
 | (2048->4096   |              | Projection    |
 |       ->256)  |              | Head          |
 |     |         |              | (2048->4096   |
 |     v         |              |       ->256)  |
 | Prediction    |              +-----+---------+
 | Head          |                    |
 | (256->4096    |                    |
 |      ->256)   |                    |
 +----+----------+                    |
      |                               |
      v                               v
   p (pred)                        z (proj)
      |                               |
      +-------> Cosine Loss <---------+
              (symmetrized)

 - No negative pairs needed (unlike contrastive methods)
 - Momentum network updated via EMA: theta_m = 0.99 * theta_m + 0.01 * theta
 - Gradient flows only through the online network
 - After pretraining, discard projection/prediction heads, keep backbone
```

---

## Hybrid Classification Model Architecture

```
 Input: 128x128x3 Image
         |
         v
 +-------+--------+
 | ResNet50        |     Backbone (initialized from Phase 1)
 | Backbone        |     Output: 2048 channels, 4x4 spatial
 | (layers 1-4)    |
 +-------+---------+
         |
         v
 (B, 2048, 4, 4)     Flatten spatial dims --> (B, 16, 2048)
         |
         v
 +-------+---------+
 | Linear Proj     |     Project 2048 -> 512
 | (2048 -> 512)   |
 +-------+---------+
         |
         v
 (B, 16, 512)
         |
   Prepend [CLS]   --> (B, 17, 512)   + Positional Embeddings
         |
         v
 +-------+---------+
 | Transformer     |     4 encoder layers
 | Encoder         |     8 attention heads
 | (d=512, h=8,    |     FFN dim = 2048
 |  L=4)           |     Dropout = 0.1
 +-------+---------+
         |
         v
   Extract [CLS]   --> (B, 512)
         |
         v
 +-------+---------+
 | Embedding FC    |     Linear(512, 512) + BatchNorm
 +-------+---------+
         |
         v
   (B, 512) embedding
         |
         v
 +-------+---------+
 | ArcFace Head    |     Angular margin: m=0.3, scale: s=30.0
 | (ArcMargin      |     Training: applies angular penalty
 |  Product)       |     Inference: cosine similarity * s
 +-------+---------+
         |
         v
   (B, num_classes) logits --> CrossEntropyLoss
```

---

## Two-Stage Training Strategy

```
 Stage A: Head-Only Training (6 epochs)
 ========================================
 - Backbone: FROZEN (no gradients)
 - Transformer Head + ArcFace: Training with LR = 1e-4
 - Purpose: Initialize the new classification layers before
            allowing backbone fine-tuning

 Stage B: Full Fine-Tuning (194 epochs, total = 200)
 =====================================================
 - Backbone: UNFROZEN, LR = 1e-5 (10x lower)
 - Transformer Head: LR = 1e-4
 - ArcFace Head: LR = 1e-4
 - Cosine annealing scheduler across Stage B
 - Purpose: Fine-tune entire model with differential learning rates
            to preserve pretrained representations
```

---

## Experiment Matrix

| Backbone Init | 1% | 5% | 10% | 25% | 50% | 100% |
|--------------|-----|-----|------|------|------|-------|
| Random | E1 | E2 | E3 | E4 | E5 | E6 |
| ImageNet | E7 | E8 | E9 | E10 | E11 | E12 |
| BYOL CASIA | E13 | E14 | E15 | E16 | E17 | E18 |
| BYOL Ekush | E19 | E20 | E21 | E22 | E23 | E24 |
| Supervised CASIA | E25 | E26 | E27 | E28 | E29 | E30 |

Each experiment produces:
- Training metrics (loss, accuracy per epoch)
- Best checkpoint (highest validation accuracy)
- Resume checkpoint (latest epoch)
- GPU/RAM usage logs

---

## Hardware Requirements

- **GPU:** NVIDIA RTX 5060 Ti (8GB VRAM) or equivalent
- **RAM:** 16GB system memory
- **Storage:** ~50GB for datasets + checkpoints
- **Optimizations applied:**
  - Mixed precision training (torch.amp)
  - Channels-last memory format
  - Gradient accumulation (effective batch = batch_size x accum_steps)
  - Persistent workers with prefetch
  - Non-blocking CUDA transfers

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
pandas>=1.4.0
tqdm>=4.62.0
Pillow>=9.0.0
lightly>=1.4.0
albumentations>=1.3.0
```

Install:
```bash
pip install -r requirements.txt
pip install lightly albumentations
```

---

## Figure Output

All figures are generated with:
- **Journal-quality styling:** serif font, clean spines, no titles (titles go in LaTeX captions)
- **Dual format:** PNG (300 DPI) + PDF (vector)
- **Consistent color palette:**
  - Blue `#1a80bb` -- BYOL CASIA (cross-script SSL)
  - Gold `#f2c45f` -- ImageNet pretrained
  - Coral `#e05a4f` -- Supervised CASIA
  - Green `#5bae6a` -- BYOL Ekush (in-domain SSL)
  - Purple `#8e6bbf` -- Combined SSL
  - Gray `#7f8c8d` -- Random initialization

---

## Key Files Reference

| File | Description |
|------|-------------|
| `src/model.py` | Model architectures: ResNet50Backbone, TransformerHead, ArcFace, HybridModel |
| `src/dataset.py` | Data loading with Albumentations augmentation pipeline |
| `src/utils.py` | Checkpointing, metrics logging, resource monitoring, seed setting |
| `src/plot_config.py` | Centralized colors, matplotlib journal style, save_figure() helper |
| `src/ssl_train.py` | BYOL self-supervised pretraining (supports CASIA / Ekush / any folder) |
| `src/supervised_pretrain.py` | Supervised classification pretraining on CASIA (3,891 classes) |
| `src/train_hybrid_small.py` | Two-stage fine-tuning with label fraction + backbone init options |
| `src/evaluate_model.py` | Evaluation: accuracy, confusion matrix, ROC, PR curves, collages |
| `src/visualize_embeddings.py` | t-SNE embedding visualization for trained models |
| `src/plot_results.py` | Cross-experiment comparison plots (label efficiency, bar charts, etc.) |
| `src/run_experiments.py` | Automated orchestration for the full 30-experiment matrix |
