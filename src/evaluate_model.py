"""
Comprehensive Evaluation Script for Hybrid Model
Provides extensive metrics including:
- Accuracy (Top-1 and Top-5)
- Precision, Recall, F1-Score (macro, micro, weighted)
- Confusion Matrix with class names
- ROC-AUC (macro/micro)
- Log Loss (Cross-Entropy)
- Matthews Correlation Coefficient (MCC)
- Specificity (per-class and average)
- Cohen's Kappa
- Per-class metrics report
- Training curves (if available)
"""

import os
import json
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import (
    DatasetFolder,
    default_loader,
    IMG_EXTENSIONS,
    make_dataset,
)
from torch.utils.data import DataLoader
import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    accuracy_score,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

from dataset import AlbumentationsImageFolder, get_transforms
from model import HybridModel, ResNet50Backbone
from utils import load_resume
from plot_config import (
    apply_journal_style, save_figure, create_figure, add_grid,
    BLUE, GOLD, CORAL, PALETTE_2, PALETTE_6, CMAP_SEQUENTIAL, get_palette,
)


# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "chkp"
CLASS_MAPPING_FILE = CHECKPOINT_DIR / "class_mapping.json"


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


def load_class_mapping(path):
    """Load class mapping from JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # Convert idx_to_class keys back to int
    mapping["idx_to_class"] = {int(k): v for k, v in mapping["idx_to_class"].items()}
    return mapping


def get_class_mapping_from_checkpoint(ckpt):
    """Extract class mapping from checkpoint if available."""
    if ckpt is None:
        return None
    class_to_idx = ckpt.get("class_to_idx")
    if class_to_idx is not None:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
        return {
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "class_names": class_names,
        }
    return None


def get_sorted_class_mapping(train_dir):
    """Create a consistent class_to_idx mapping using NUMERICAL sorting for numeric folder names."""
    class_names = sorted(os.listdir(train_dir))

    # Check if all class names are numeric
    all_numeric = all(name.isdigit() for name in class_names)

    if all_numeric:
        class_names = sorted(class_names, key=lambda x: int(x))
        print(f"[INFO] Detected numeric class names. Using numerical sorting.")
    else:
        print(f"[INFO] Using alphabetical sorting for class names.")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    return {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_names": class_names,
    }


def build_loader(data_root, split, input_size, batch_size, class_to_idx, num_workers=0):
    """Build data loader with fixed class mapping."""
    _, val_tf = get_transforms(input_size)

    folder = FixedClassImageFolder(os.path.join(data_root, split), class_to_idx)
    ds = AlbumentationsImageFolder(folder, transform=val_tf)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print(f"[INFO] Split '{split}' samples: {len(folder)}")
    return loader


def load_model(model_path, num_classes, device, input_size=128):
    """Load model from checkpoint."""
    ckpt = load_resume(model_path, device)
    if ckpt is None:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"[INFO] Checkpoint keys: {list(ckpt.keys())}")
    print(f"[INFO] Checkpoint best_acc: {ckpt.get('best_acc', 'N/A')}")
    print(f"[INFO] Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"[INFO] Checkpoint stage: {ckpt.get('stage', 'N/A')}")

    if "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt

    resnet_backbone = ResNet50Backbone()
    model = HybridModel(
        num_classes=num_classes,
        backbone=resnet_backbone,
        backbone_channels=2048,
        input_size=input_size,
    )

    try:
        model.load_state_dict(state)
        print("[INFO] Model loaded successfully (strict=True)")
    except Exception as e:
        print(f"[WARN] Strict load failed: {e}. Trying non-strict load.")
        model.load_state_dict(state, strict=False)
        print("[INFO] Model loaded with strict=False")

    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_inference(model, loader, device):
    """Run inference and collect predictions."""
    y_true = []
    y_pred = []
    y_prob = []
    y_logits = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.append(probs.cpu().numpy())
        y_logits.append(logits.cpu().numpy())

    y_prob = np.concatenate(y_prob, axis=0)
    y_logits = np.concatenate(y_logits, axis=0)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"[INFO] Total samples: {len(y_true)}")
    print(
        f"[INFO] Unique y_true: {len(np.unique(y_true))}, range: [{y_true.min()}, {y_true.max()}]"
    )
    print(
        f"[INFO] Unique y_pred: {len(np.unique(y_pred))}, range: [{y_pred.min()}, {y_pred.max()}]"
    )

    return y_true, y_pred, y_prob, y_logits


def compute_topk_accuracy(y_prob, y_true, k=5):
    """Compute top-k accuracy."""
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return correct.mean()


def compute_specificity(y_true, y_pred, num_classes):
    """Compute per-class specificity (True Negative Rate)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificity_per_class = []

    for i in range(num_classes):
        # True negatives: sum of all elements except row i and column i
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        # False positives: sum of column i except diagonal
        fp = cm[:, i].sum() - cm[i, i]

        if (tn + fp) > 0:
            specificity_per_class.append(tn / (tn + fp))
        else:
            specificity_per_class.append(0.0)

    return np.array(specificity_per_class)


def compute_all_metrics(y_true, y_pred, y_prob, class_names):
    """Compute all evaluation metrics."""
    num_classes = len(class_names)
    metrics = {}

    # Basic accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["top1_accuracy"] = metrics["accuracy"]
    metrics["top5_accuracy"] = compute_topk_accuracy(
        y_prob, y_true, k=min(5, num_classes)
    )

    # Precision, Recall, F1 (macro, micro, weighted)
    for avg in ["macro", "micro", "weighted"]:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"precision_{avg}"] = prec
        metrics[f"recall_{avg}"] = rec
        metrics[f"f1_{avg}"] = f1

    # Log Loss (Cross-Entropy)
    try:
        # Clip probabilities to avoid log(0)
        y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
        metrics["log_loss"] = log_loss(
            y_true, y_prob_clipped, labels=list(range(num_classes))
        )
    except Exception as e:
        print(f"[WARN] Could not compute log_loss: {e}")
        metrics["log_loss"] = None

    # Matthews Correlation Coefficient
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # Cohen's Kappa
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    # Specificity
    specificity = compute_specificity(y_true, y_pred, num_classes)
    metrics["specificity_macro"] = specificity.mean()
    metrics["specificity_per_class"] = specificity.tolist()

    # ROC-AUC
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        metrics["roc_auc_macro"] = roc_auc_score(
            y_true_bin, y_prob, average="macro", multi_class="ovr"
        )
        metrics["roc_auc_weighted"] = roc_auc_score(
            y_true_bin, y_prob, average="weighted", multi_class="ovr"
        )
    except Exception as e:
        print(f"[WARN] Could not compute ROC-AUC: {e}")
        metrics["roc_auc_macro"] = None
        metrics["roc_auc_weighted"] = None

    # Average Precision Score (macro)
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        metrics["average_precision_macro"] = average_precision_score(
            y_true_bin, y_prob, average="macro"
        )
    except Exception as e:
        print(f"[WARN] Could not compute average precision: {e}")
        metrics["average_precision_macro"] = None

    return metrics


def get_per_class_metrics(y_true, y_pred, y_prob, class_names):
    """Get detailed per-class metrics."""
    num_classes = len(class_names)

    # Classification report dict
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    # Add specificity per class
    specificity = compute_specificity(y_true, y_pred, num_classes)
    for i, name in enumerate(class_names):
        if name in report:
            report[name]["specificity"] = specificity[i]

    return report


def plot_confusion_matrix(cm, class_names, output_path, normalize=True):
    """Plot and save confusion matrix (journal quality, no title)."""
    apply_journal_style()
    num_classes = len(class_names)

    if normalize:
        cm_plot = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    figsize = max(8, num_classes * 0.25)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))

    if num_classes <= 30:
        sns.heatmap(
            cm_plot, annot=True, fmt=fmt, cmap=CMAP_SEQUENTIAL,
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, annot_kws={"size": 6}, linewidths=0.3, linecolor="#eeeeee",
        )
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
    else:
        im = ax.imshow(cm_plot, interpolation="nearest", cmap=CMAP_SEQUENTIAL)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        step = max(1, num_classes // 20)
        ticks = np.arange(0, num_classes, step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([class_names[i] for i in ticks], rotation=90, fontsize=6)
        ax.set_yticklabels([class_names[i] for i in ticks], fontsize=6)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_roc_curves(y_true, y_prob, class_names, output_path):
    """Plot ROC curves (journal quality, no title)."""
    apply_journal_style()
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(6, 5))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        except Exception:
            fpr[i] = np.array([0, 1])
            tpr[i] = np.array([0, 1])
            roc_auc[i] = 0.5

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    macro_auc = np.mean(list(roc_auc.values()))

    if num_classes <= 20:
        colors = get_palette(num_classes)
        for i, color in zip(range(num_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=1, alpha=0.7,
                    label=f"{class_names[i]} ({roc_auc[i]:.2f})")
    else:
        for i in range(num_classes):
            ax.plot(fpr[i], tpr[i], color="#cccccc", lw=0.4, alpha=0.3)
        sorted_auc = sorted(roc_auc.items(), key=lambda x: x[1])
        for idx, color, lbl in [(sorted_auc[0][0], CORAL, "Worst"),
                                 (sorted_auc[len(sorted_auc)//2][0], GOLD, "Median"),
                                 (sorted_auc[-1][0], BLUE, "Best")]:
            ax.plot(fpr[idx], tpr[idx], color=color, lw=1.8,
                    label=f"{lbl}: {class_names[idx]} ({roc_auc[idx]:.2f})")

    ax.plot(all_fpr, mean_tpr, label=f"Macro-avg (AUC={macro_auc:.3f})",
            color=BLUE, linewidth=2, linestyle="--")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=0.8, linestyle=":")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=7)
    add_grid(ax, axis="both", alpha=0.2)

    plt.tight_layout()
    save_figure(fig, output_path)
    return macro_auc, roc_auc


def plot_precision_recall_curves(y_true, y_prob, class_names, output_path):
    """Plot Precision-Recall curves (journal quality, no title)."""
    apply_journal_style()
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(6, 5))

    precision_d = dict()
    recall_d = dict()
    ap = dict()
    for i in range(num_classes):
        try:
            precision_d[i], recall_d[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            ap[i] = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        except Exception:
            precision_d[i] = np.array([0, 1])
            recall_d[i] = np.array([1, 0])
            ap[i] = 0.0

    macro_ap = np.mean(list(ap.values()))
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
    ap_micro = average_precision_score(y_true_bin, y_prob, average="micro")

    if num_classes <= 20:
        colors = get_palette(num_classes)
        for i, color in zip(range(num_classes), colors):
            ax.plot(recall_d[i], precision_d[i], color=color, lw=1, alpha=0.7,
                    label=f"{class_names[i]} ({ap[i]:.2f})")
    else:
        for i in range(num_classes):
            ax.plot(recall_d[i], precision_d[i], color="#cccccc", lw=0.4, alpha=0.3)
        sorted_ap = sorted(ap.items(), key=lambda x: x[1])
        for idx, color, lbl in [(sorted_ap[0][0], CORAL, "Worst"),
                                 (sorted_ap[len(sorted_ap)//2][0], GOLD, "Median"),
                                 (sorted_ap[-1][0], BLUE, "Best")]:
            ax.plot(recall_d[idx], precision_d[idx], color=color, lw=1.8,
                    label=f"{lbl}: {class_names[idx]} ({ap[idx]:.2f})")

    ax.plot(recall_micro, precision_micro, color=BLUE, lw=2, linestyle="--",
            label=f"Micro-avg (AP={ap_micro:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=7)
    add_grid(ax, axis="both", alpha=0.2)

    plt.tight_layout()
    save_figure(fig, output_path)
    return macro_ap


def plot_class_accuracy_bar(report, class_names, output_path):
    """Plot per-class Precision/Recall/F1 bar chart (journal quality, no title)."""
    apply_journal_style()
    f1_scores = [report.get(name, {}).get("f1-score", 0) for name in class_names]
    precisions = [report.get(name, {}).get("precision", 0) for name in class_names]
    recalls = [report.get(name, {}).get("recall", 0) for name in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.25), 4))

    ax.bar(x - width, precisions, width, label="Precision", color=BLUE, alpha=0.85)
    ax.bar(x, recalls, width, label="Recall", color=GOLD, alpha=0.85)
    ax.bar(x + width, f1_scores, width, label="F1-Score", color=CORAL, alpha=0.85)

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.legend(fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=np.mean(f1_scores), color=CORAL, linestyle="--", lw=0.8, alpha=0.6)
    add_grid(ax, alpha=0.2)

    plt.tight_layout()
    save_figure(fig, output_path)


def make_prediction_collage(
    model, folder, transform, device, output_path, num_samples=16, seed=42
):
    """Create a collage of sample predictions."""
    rng = random.Random(seed)
    samples = folder.imgs
    if len(samples) == 0:
        return

    picks = rng.sample(samples, k=min(num_samples, len(samples)))

    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    model.eval()
    with torch.no_grad():
        for ax, (path, true_label) in zip(axes, picks):
            img_pil = Image.open(path).convert("RGB")
            arr = np.array(img_pil)
            tensor = transform(image=arr)["image"]
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            tensor = tensor.unsqueeze(0).to(device)

            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            pred = int(probs.argmax(1).item())
            prob = float(probs.max().item())

            ax.imshow(img_pil)
            ax.axis("off")
            color = "green" if pred == true_label else "red"
            ax.set_title(
                f"True:{true_label} Pred:{pred}\nConf:{prob:.2f}",
                fontsize=8,
                color=color,
            )

    # Blank out unused axes
    for ax in axes[len(picks) :]:
        ax.axis("off")

    plt.tight_layout()
    save_figure(fig, output_path)


def print_summary(metrics, split):
    """Print a formatted summary of metrics."""
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY - {split.upper()}")
    print("=" * 60)

    print(f"\n{'ACCURACY METRICS':=^50}")
    print(
        f"  Top-1 Accuracy:       {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)"
    )
    print(
        f"  Top-5 Accuracy:       {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)"
    )

    print(f"\n{'PRECISION/RECALL/F1 (Macro)':=^50}")
    print(f"  Precision (macro):    {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):       {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (macro):     {metrics['f1_macro']:.4f}")

    print(f"\n{'PRECISION/RECALL/F1 (Micro)':=^50}")
    print(f"  Precision (micro):    {metrics['precision_micro']:.4f}")
    print(f"  Recall (micro):       {metrics['recall_micro']:.4f}")
    print(f"  F1-Score (micro):     {metrics['f1_micro']:.4f}")

    print(f"\n{'PRECISION/RECALL/F1 (Weighted)':=^50}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (weighted):  {metrics['f1_weighted']:.4f}")

    print(f"\n{'OTHER METRICS':=^50}")
    if metrics.get("log_loss") is not None:
        print(f"  Log Loss (CE):        {metrics['log_loss']:.4f}")
    if metrics.get("roc_auc_macro") is not None:
        print(f"  ROC-AUC (macro):      {metrics['roc_auc_macro']:.4f}")
    if metrics.get("roc_auc_weighted") is not None:
        print(f"  ROC-AUC (weighted):   {metrics['roc_auc_weighted']:.4f}")
    if metrics.get("average_precision_macro") is not None:
        print(f"  Avg Precision (macro):{metrics['average_precision_macro']:.4f}")
    print(f"  MCC:                  {metrics['mcc']:.4f}")
    print(f"  Cohen's Kappa:        {metrics['cohen_kappa']:.4f}")
    print(f"  Specificity (macro):  {metrics['specificity_macro']:.4f}")

    print("=" * 60 + "\n")


def evaluate(
    model_path,
    data_root,
    split="test",
    input_size=128,
    batch_size=64,
    num_classes=50,
    output_dir=None,
):
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Model path: {model_path}")
    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Split: {split}")

    # Load model and checkpoint
    model, ckpt = load_model(
        model_path, num_classes=num_classes, device=device, input_size=input_size
    )

    # Get class mapping - priority: checkpoint > JSON file > from train folder
    mapping = get_class_mapping_from_checkpoint(ckpt)
    if mapping is None:
        mapping = load_class_mapping(CLASS_MAPPING_FILE)
    if mapping is None:
        train_dir = os.path.join(data_root, "train")
        mapping = get_sorted_class_mapping(train_dir)
        print(f"[INFO] Created class mapping from train folder")
    else:
        print(
            f"[INFO] Loaded class mapping (source: {'checkpoint' if 'class_to_idx' in ckpt else 'JSON file'})"
        )

    class_to_idx = mapping["class_to_idx"]
    class_names = mapping["class_names"]

    print(f"[INFO] Number of classes: {len(class_names)}")
    print(f"[INFO] Class names (first 5): {class_names[:5]}")
    print(f"[INFO] Class names (last 5): {class_names[-5:]}")

    # Build data loader
    loader = build_loader(data_root, split, input_size, batch_size, class_to_idx)

    # Run inference
    y_true, y_pred, y_prob, y_logits = run_inference(model, loader, device)

    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)
    per_class_report = get_per_class_metrics(y_true, y_pred, y_prob, class_names)

    # Print summary
    print_summary(metrics, split)

    # Output directory
    out_dir = (
        Path(output_dir) if output_dir else PROJECT_ROOT / "reports" / f"eval_{split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_saveable = {
        k: (v if not isinstance(v, np.ndarray) else v.tolist())
        for k, v in metrics.items()
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_saveable, f, indent=2)
    print(f"[INFO] Saved metrics: {out_dir / 'metrics.json'}")

    # Save classification report
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(per_class_report, f, indent=2)
    print(
        f"[INFO] Saved classification report: {out_dir / 'classification_report.json'}"
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion_matrix(
        cm, class_names, out_dir / "confusion_matrix.png", normalize=True
    )
    plot_confusion_matrix(
        cm, class_names, out_dir / "confusion_matrix_raw.png", normalize=False
    )

    # Save confusion matrix as CSV
    import pandas as pd

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    print(f"[INFO] Saved confusion matrix CSV: {out_dir / 'confusion_matrix.csv'}")

    # ROC curves
    if y_prob.ndim == 2:
        plot_roc_curves(y_true, y_prob, class_names, out_dir / "roc_curves.png")
        plot_precision_recall_curves(
            y_true, y_prob, class_names, out_dir / "pr_curves.png"
        )

    # Per-class metrics bar chart
    plot_class_accuracy_bar(
        per_class_report, class_names, out_dir / "per_class_metrics.png"
    )

    # Prediction collage
    _, val_tf = get_transforms(input_size)
    folder = FixedClassImageFolder(os.path.join(data_root, split), class_to_idx)
    make_prediction_collage(
        model,
        folder,
        val_tf,
        device,
        out_dir / "prediction_collage.png",
        num_samples=16,
    )

    print(f"\n[INFO] All artifacts saved to: {out_dir}")

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation for hybrid model"
    )
    parser.add_argument("--model", required=True, help="Path to checkpoint (best-val)")
    parser.add_argument(
        "--data_root",
        default=str(PROJECT_ROOT / "data"),
        help="Dataset root containing train/val/test folders",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=50)
    parser.add_argument(
        "--output_dir", default=None, help="Where to save metrics/plots"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model,
        data_root=args.data_root,
        split=args.split,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
    )
