"""
Generate all comparative figures across experiments for the journal paper.

Figures generated:
  1. Label efficiency curves (accuracy vs label fraction, one line per pretraining source)
  2. Pretraining source comparison bar chart (at 100% labels)
  3. Training loss curves overlay (from training_metrics.json)
  4. Training accuracy curves overlay
  5. Validation accuracy curves overlay
  6. Per-class accuracy delta heatmap (cross-script vs baseline)
  7. Label savings summary (how much data each method saves)
  8. Learning rate schedule visualization
  9. SSL pretraining loss curves
 10. Bar chart: best accuracy per experiment

Usage:
  python plot_results.py --experiment_dir ../chkp/experiments --output_dir ../reports/figures
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_config import (
    apply_journal_style, save_figure, create_figure, add_grid,
    BLUE, GOLD, CORAL, GREEN, PURPLE, GRAY,
    PALETTE_2, PALETTE_6, EXPERIMENT_COLORS, FRACTION_MARKERS,
    get_experiment_color, get_palette,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_experiment_results(experiment_dir):
    """Load all experiment results from checkpoint directories."""
    import torch
    results = []
    exp_dir = Path(experiment_dir)
    if not exp_dir.exists():
        return results

    for sub in sorted(exp_dir.iterdir()):
        best_ckpt = sub / "best_checkpoint.pth"
        metrics_file = sub / "training_metrics.json"
        if not best_ckpt.exists():
            continue

        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)

        history = []
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    history = json.load(f)
            except Exception:
                pass

        results.append({
            "name": sub.name,
            "backbone_init": ckpt.get("backbone_init", "unknown"),
            "label_fraction": ckpt.get("label_fraction", 1.0),
            "best_val_acc": ckpt.get("best_acc", 0.0),
            "epoch": ckpt.get("epoch", -1),
            "history": history,
        })
    return results


def load_ssl_metrics(ssl_ckpt_dir):
    """Load SSL pretraining metrics."""
    metrics = {}
    ssl_dir = Path(ssl_ckpt_dir)
    if not ssl_dir.exists():
        return metrics
    for f in ssl_dir.glob("*_metrics.json"):
        name = f.stem.replace("_metrics", "")
        try:
            with open(f) as fh:
                metrics[name] = json.load(fh)
        except Exception:
            pass
    return metrics


# ============================================================
# FIGURE 1: Label Efficiency Curves
# ============================================================

def plot_label_efficiency(results, output_path):
    """Accuracy vs label fraction, one line per pretraining source."""
    apply_journal_style()

    # Group by backbone_init
    groups = {}
    for r in results:
        key = r["backbone_init"]
        # Refine key from experiment name
        for prefix in ["ssl_casia", "ssl_ekush", "supervised_casia", "ssl_combined", "imagenet", "random"]:
            if prefix in r["name"]:
                key = prefix
                break
        if key not in groups:
            groups[key] = []
        groups[key].append((r["label_fraction"], r["best_val_acc"]))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for name, points in sorted(groups.items()):
        points.sort(key=lambda x: x[0])
        fracs = [p[0] for p in points]
        accs = [p[1] for p in points]
        color = get_experiment_color(name)
        ax.plot(fracs, accs, marker="o", color=color, label=name.replace("_", " ").title(),
                linewidth=1.8, markersize=5)

    ax.set_xlabel("Label Fraction")
    ax.set_ylabel("Validation Accuracy")
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}" if x >= 0.1 else f"{x:.0%}"))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="lower right")
    add_grid(ax, alpha=0.2)

    plt.tight_layout()
    save_figure(fig, output_path)


# ============================================================
# FIGURE 2: Pretraining Source Comparison Bar Chart
# ============================================================

def plot_pretraining_comparison_bar(results, output_path, target_fraction=1.0):
    """Bar chart comparing pretraining sources at a fixed label fraction."""
    apply_journal_style()

    # Filter to target fraction
    filtered = [r for r in results if abs(r["label_fraction"] - target_fraction) < 0.001]
    if not filtered:
        print(f"[WARN] No results at fraction={target_fraction}")
        return

    names = []
    accs = []
    colors = []
    for r in sorted(filtered, key=lambda x: x["best_val_acc"]):
        for prefix in ["ssl_casia", "ssl_ekush", "supervised_casia", "ssl_combined", "imagenet", "random"]:
            if prefix in r["name"]:
                names.append(prefix.replace("_", " ").title())
                colors.append(get_experiment_color(prefix))
                break
        else:
            names.append(r["name"])
            colors.append(GRAY)
        accs.append(r["best_val_acc"])

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(range(len(names)), accs, color=colors, height=0.6, edgecolor="white", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{acc:.2%}", va="center", fontsize=8)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Validation Accuracy")
    ax.set_xlim(right=max(accs) * 1.12)
    add_grid(ax, axis="x", alpha=0.2)

    plt.tight_layout()
    save_figure(fig, output_path)


# ============================================================
# FIGURE 3-5: Training Curves (Loss, Train Acc, Val Acc)
# ============================================================

def plot_training_curves(results, output_dir, target_fraction=1.0):
    """Overlay training curves for experiments at a given label fraction."""
    apply_journal_style()
    filtered = [r for r in results if abs(r["label_fraction"] - target_fraction) < 0.001 and r["history"]]
    if not filtered:
        return

    # Loss curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in filtered:
        epochs = [h["epoch"] for h in r["history"] if "train_loss" in h]
        losses = [h["train_loss"] for h in r["history"] if "train_loss" in h]
        if epochs:
            name = r["name"].split("_frac")[0]
            ax.plot(epochs, losses, color=get_experiment_color(name), label=name.replace("_"," ").title(), lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.legend(fontsize=7)
    add_grid(ax, alpha=0.2)
    plt.tight_layout()
    save_figure(fig, output_dir / "training_loss_curves")

    # Val accuracy curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in filtered:
        epochs = [h["epoch"] for h in r["history"] if "val_acc" in h]
        vals = [h["val_acc"] for h in r["history"] if "val_acc" in h]
        if epochs:
            name = r["name"].split("_frac")[0]
            ax.plot(epochs, vals, color=get_experiment_color(name), label=name.replace("_"," ").title(), lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.legend(fontsize=7, loc="lower right")
    add_grid(ax, alpha=0.2)
    plt.tight_layout()
    save_figure(fig, output_dir / "val_accuracy_curves")

    # Train accuracy curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in filtered:
        epochs = [h["epoch"] for h in r["history"] if "train_acc" in h]
        taccs = [h["train_acc"] for h in r["history"] if "train_acc" in h]
        if epochs:
            name = r["name"].split("_frac")[0]
            ax.plot(epochs, taccs, color=get_experiment_color(name), label=name.replace("_"," ").title(), lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Accuracy")
    ax.legend(fontsize=7, loc="lower right")
    add_grid(ax, alpha=0.2)
    plt.tight_layout()
    save_figure(fig, output_dir / "train_accuracy_curves")


# ============================================================
# FIGURE 6: SSL Pretraining Loss Curves
# ============================================================

def plot_ssl_loss_curves(ssl_metrics, output_path):
    """Overlay SSL pretraining loss curves."""
    apply_journal_style()
    if not ssl_metrics:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    color_map = {"byol_casia": BLUE, "byol_ekush": GREEN}

    for name, history in ssl_metrics.items():
        epochs = [h["epoch"] for h in history if "train_loss" in h]
        losses = [h["train_loss"] for h in history if "train_loss" in h]
        if epochs:
            display = name.replace("byol_", "BYOL-").replace("_", " ").title()
            ax.plot(epochs, losses, color=color_map.get(name, GRAY), label=display, lw=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("BYOL Loss")
    ax.legend(fontsize=8)
    add_grid(ax, alpha=0.2)
    plt.tight_layout()
    save_figure(fig, output_path)


# ============================================================
# FIGURE 7: Label Savings Summary
# ============================================================

def plot_label_savings(results, output_path, baseline_name="random"):
    """Show how much labeled data each pretraining method saves to reach baseline 100% accuracy."""
    apply_journal_style()

    # Get baseline accuracy at 100% labels
    baseline_100 = None
    for r in results:
        if baseline_name in r["name"] and abs(r["label_fraction"] - 1.0) < 0.001:
            baseline_100 = r["best_val_acc"]
            break

    if baseline_100 is None:
        print("[WARN] No baseline found for label savings plot")
        return

    # For each method, find the lowest fraction that reaches baseline_100
    groups = {}
    for r in results:
        for prefix in ["ssl_casia", "ssl_ekush", "supervised_casia", "imagenet"]:
            if prefix in r["name"]:
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append((r["label_fraction"], r["best_val_acc"]))
                break

    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = []
    savings = []
    colors = []

    for method, points in sorted(groups.items()):
        points.sort(key=lambda x: x[0])
        # Find minimum fraction that achieves baseline accuracy
        min_frac = 1.0
        for frac, acc in points:
            if acc >= baseline_100 * 0.99:  # within 1% of baseline
                min_frac = frac
                break
        names.append(method.replace("_", " ").title())
        savings.append((1.0 - min_frac) * 100)
        colors.append(get_experiment_color(method))

    bars = ax.barh(range(len(names)), savings, color=colors, height=0.55)
    for bar, s in zip(bars, savings):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{s:.0f}%", va="center", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Label Savings (%)")
    ax.set_xlim(right=105)
    add_grid(ax, axis="x", alpha=0.2)
    plt.tight_layout()
    save_figure(fig, output_path)


# ============================================================
# FIGURE 8: Summary Table as Figure
# ============================================================

def plot_results_table(results, output_path):
    """Generate a summary results table as a figure."""
    apply_journal_style()

    # Pivot: rows = methods, columns = fractions
    groups = {}
    for r in results:
        for prefix in ["ssl_casia", "ssl_ekush", "supervised_casia", "imagenet", "random"]:
            if prefix in r["name"]:
                if prefix not in groups:
                    groups[prefix] = {}
                groups[prefix][r["label_fraction"]] = r["best_val_acc"]
                break

    if not groups:
        return

    methods = sorted(groups.keys())
    fractions = sorted(set(f for g in groups.values() for f in g.keys()))

    cell_text = []
    for m in methods:
        row = []
        for f in fractions:
            val = groups[m].get(f, None)
            row.append(f"{val:.2%}" if val is not None else "-")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(8, 0.5 + len(methods) * 0.4))
    ax.axis("off")

    col_labels = [f"{f:.0%}" for f in fractions]
    row_labels = [m.replace("_", " ").title() for m in methods]

    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(fractions)):
        table[0, j].set_facecolor(BLUE)
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(methods)):
        table[i+1, -1].set_facecolor("#f5f5f5")

    plt.tight_layout()
    save_figure(fig, output_path)


# ============================================================
# FIGURE 9: GPU/RAM Usage Over Training
# ============================================================

def plot_resource_usage(results, output_path):
    """Plot GPU VRAM and RAM usage over training epochs."""
    apply_journal_style()
    # Pick one experiment with history
    example = None
    for r in results:
        if r["history"] and any("gpu_allocated_mb" in h for h in r["history"]):
            example = r
            break
    if example is None:
        return

    history = example["history"]
    epochs = [h["epoch"] for h in history if "gpu_allocated_mb" in h]
    vram = [h["gpu_allocated_mb"] for h in history if "gpu_allocated_mb" in h]
    ram = [h.get("ram_used_mb", 0) for h in history if "gpu_allocated_mb" in h]

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax2 = ax1.twinx()

    ax1.plot(epochs, vram, color=BLUE, lw=1.5, label="GPU VRAM (MB)")
    ax2.plot(epochs, [r/1024 for r in ram], color=GOLD, lw=1.5, label="System RAM (GB)")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("GPU VRAM (MB)", color=BLUE)
    ax2.set_ylabel("System RAM (GB)", color=GOLD)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2.tick_params(axis="y", labelcolor=GOLD)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    plt.tight_layout()
    save_figure(fig, output_path)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate all comparative figures")
    parser.add_argument("--experiment_dir", type=str, default=str(PROJECT_ROOT / "chkp" / "experiments"))
    parser.add_argument("--ssl_dir", type=str, default=str(PROJECT_ROOT / "chkp" / "ssl_pretrain"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "reports" / "figures"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiment results...")
    results = load_experiment_results(args.experiment_dir)
    ssl_metrics = load_ssl_metrics(args.ssl_dir)

    if not results:
        print("No experiment results found. Run experiments first.")
        return

    print(f"Found {len(results)} experiments, {len(ssl_metrics)} SSL runs")

    # Generate all figures
    print("\nGenerating figures...")

    plot_label_efficiency(results, output_dir / "label_efficiency")
    plot_pretraining_comparison_bar(results, output_dir / "pretraining_comparison_100pct")
    plot_pretraining_comparison_bar(results, output_dir / "pretraining_comparison_10pct", target_fraction=0.10)
    plot_pretraining_comparison_bar(results, output_dir / "pretraining_comparison_5pct", target_fraction=0.05)
    plot_training_curves(results, output_dir, target_fraction=1.0)
    plot_ssl_loss_curves(ssl_metrics, output_dir / "ssl_loss_curves")
    plot_label_savings(results, output_dir / "label_savings")
    plot_results_table(results, output_dir / "results_table")
    plot_resource_usage(results, output_dir / "resource_usage")

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
