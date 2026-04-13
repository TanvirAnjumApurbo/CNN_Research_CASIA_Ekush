"""
Experiment orchestration for Cross-Script Transfer Learning study.

Runs the full experimental matrix:
  - Pretraining sources: random, imagenet, ssl_casia, ssl_ekush, supervised_casia, ssl_combined
  - Label fractions: 1%, 5%, 10%, 25%, 50%, 100%

Usage:
  # Run all experiments sequentially
  python run_experiments.py --phase all

  # Run only SSL pretraining
  python run_experiments.py --phase pretrain

  # Run only fine-tuning experiments
  python run_experiments.py --phase finetune

  # Run a single specific experiment
  python run_experiments.py --phase single --backbone_init ssl --ssl_checkpoint path/to/ckpt --label_fraction 0.1
"""

import os
import subprocess
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Dataset paths (absolute paths on this machine)
EKUSH_ROOT = Path("C:/Ekush/dataset/dataset")
CASIA_ROOT = Path("C:/CASIA")

# Checkpoint directories for SSL pretraining
SSL_CKPT_DIR = PROJECT_ROOT / "chkp" / "ssl_pretrain"
SUPERVISED_CKPT_DIR = PROJECT_ROOT / "chkp" / "supervised_pretrain"
EXPERIMENT_DIR = PROJECT_ROOT / "chkp" / "experiments"

LABEL_FRACTIONS = [0.01, 0.10, 1.0]


def run_cmd(cmd, description=""):
    """Run a command and print output in real-time."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"TIME: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(SRC_DIR))
    if result.returncode != 0:
        print(f"[WARN] Command failed with return code {result.returncode}")
    return result.returncode


def phase_pretrain(args):
    """Run all pretraining jobs."""
    os.makedirs(SSL_CKPT_DIR, exist_ok=True)

    # 1. BYOL on CASIA (cross-script)
    casia_ssl_ckpt = SSL_CKPT_DIR / "byol_casia.pth"
    if not casia_ssl_ckpt.exists() or args.force:
        run_cmd([
            sys.executable, str(SRC_DIR / "ssl_train.py"),
            "--data_root", str(CASIA_ROOT),
            "--checkpoint_path", str(casia_ssl_ckpt),
            "--batch_size", "192",
            "--grad_accum_steps", "1",
            "--epochs", "100",
            "--backbone", "resnet50",
            "--subsample", "0.25",
            "--num_workers", "0",
        ], "BYOL SSL Pretraining on CASIA (Chinese)")
    else:
        print(f"[SKIP] CASIA SSL checkpoint exists: {casia_ssl_ckpt}")

    # 2. BYOL on Ekush (in-domain)
    ekush_ssl_ckpt = SSL_CKPT_DIR / "byol_ekush.pth"
    if not ekush_ssl_ckpt.exists() or args.force:
        run_cmd([
            sys.executable, str(SRC_DIR / "ssl_train.py"),
            "--data_root", str(EKUSH_ROOT),
            "--checkpoint_path", str(ekush_ssl_ckpt),
            "--batch_size", "192",
            "--grad_accum_steps", "1",
            "--epochs", "100",
            "--backbone", "resnet50",
            "--subsample", "0.5",
            "--num_workers", "0",
        ], "BYOL SSL Pretraining on Ekush (Bangla)")
    else:
        print(f"[SKIP] Ekush SSL checkpoint exists: {ekush_ssl_ckpt}")

    # 3. Supervised pretraining on CASIA
    if not (SUPERVISED_CKPT_DIR / "backbone_weights.pth").exists() or args.force:
        run_cmd([
            sys.executable, str(SRC_DIR / "supervised_pretrain.py"),
            "--data_root", str(CASIA_ROOT),
            "--checkpoint_dir", str(SUPERVISED_CKPT_DIR),
            "--epochs", "50",
            "--batch_size", "64",
            "--lr", "0.001",
        ], "Supervised Pretraining on CASIA (3891 classes)")
    else:
        print(f"[SKIP] Supervised pretrain checkpoint exists")


def phase_finetune(args):
    """Run fine-tuning experiments on Ekush with different backbone inits and label fractions."""
    ssl_casia_ckpt = SSL_CKPT_DIR / "byol_casia.pth"
    ssl_ekush_ckpt = SSL_CKPT_DIR / "byol_ekush.pth"
    sup_casia_ckpt = SUPERVISED_CKPT_DIR / "backbone_weights.pth"

    # Define experiment configurations
    experiments = [
        {"name": "random", "backbone_init": "random", "ssl_checkpoint": None},
        {"name": "imagenet", "backbone_init": "imagenet", "ssl_checkpoint": None},
    ]

    # Add SSL experiments only if checkpoints exist
    if ssl_casia_ckpt.exists():
        experiments.append({"name": "ssl_casia", "backbone_init": "ssl",
                            "ssl_checkpoint": str(ssl_casia_ckpt)})
    else:
        print(f"[WARN] CASIA SSL checkpoint not found, skipping ssl_casia experiments")

    if ssl_ekush_ckpt.exists():
        experiments.append({"name": "ssl_ekush", "backbone_init": "ssl",
                            "ssl_checkpoint": str(ssl_ekush_ckpt)})
    else:
        print(f"[WARN] Ekush SSL checkpoint not found, skipping ssl_ekush experiments")

    if sup_casia_ckpt.exists():
        experiments.append({"name": "supervised_casia", "backbone_init": "ssl",
                            "ssl_checkpoint": str(sup_casia_ckpt)})
    else:
        print(f"[WARN] Supervised pretrain checkpoint not found, skipping supervised_casia experiments")

    fractions = args.label_fractions if args.label_fractions else LABEL_FRACTIONS

    # Track results
    results = {}

    for exp in experiments:
        for frac in fractions:
            exp_name = f"{exp['name']}_frac{frac:.2f}"
            exp_dir = EXPERIMENT_DIR / exp_name

            # Skip if already completed
            best_ckpt = exp_dir / "best_checkpoint.pth"
            if best_ckpt.exists() and not args.force:
                print(f"[SKIP] {exp_name} already completed")
                continue

            cmd = [
                sys.executable, str(SRC_DIR / "train_hybrid_small.py"),
                "--data_root", str(EKUSH_ROOT),
                "--backbone_init", exp["backbone_init"],
                "--label_fraction", str(frac),
                "--experiment_name", exp_name,
                "--checkpoint_dir", str(exp_dir),
                "--batch_size", "128",
                "--eval_batch_size", "128",
            ]

            if exp["ssl_checkpoint"]:
                cmd.extend(["--ssl_checkpoint", exp["ssl_checkpoint"]])

            desc = f"Fine-tune: {exp['name']} | label_fraction={frac}"
            ret = run_cmd(cmd, desc)

            results[exp_name] = {"return_code": ret, "backbone": exp["name"], "fraction": frac}

    # Save results summary
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    with open(EXPERIMENT_DIR / "run_log.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRun log saved to: {EXPERIMENT_DIR / 'run_log.json'}")


def phase_visualize(args):
    """Generate t-SNE plots for all completed experiments."""
    if not EXPERIMENT_DIR.exists():
        print("No experiments found. Run finetune phase first.")
        return

    for exp_dir in sorted(EXPERIMENT_DIR.iterdir()):
        best_ckpt = exp_dir / "best_checkpoint.pth"
        if not best_ckpt.exists():
            continue

        output_dir = PROJECT_ROOT / "reports" / "embeddings" / exp_dir.name
        if (output_dir / f"tsne_{exp_dir.name}.png").exists() and not args.force:
            print(f"[SKIP] {exp_dir.name} visualization exists")
            continue

        run_cmd([
            sys.executable, str(SRC_DIR / "visualize_embeddings.py"),
            "--model_path", str(best_ckpt),
            "--data_root", str(EKUSH_ROOT),
            "--split", "test",
            "--max_samples", "3000",
            "--output_dir", str(output_dir),
            "--experiment_name", exp_dir.name,
        ], f"t-SNE visualization: {exp_dir.name}")


def phase_collect_results(args):
    """Collect all experiment results into a summary table."""
    if not EXPERIMENT_DIR.exists():
        print("No experiments found.")
        return

    import torch
    results = []

    for exp_dir in sorted(EXPERIMENT_DIR.iterdir()):
        best_ckpt = exp_dir / "best_checkpoint.pth"
        if not best_ckpt.exists():
            continue

        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        results.append({
            "experiment": exp_dir.name,
            "backbone_init": ckpt.get("backbone_init", "unknown"),
            "label_fraction": ckpt.get("label_fraction", 1.0),
            "best_val_acc": ckpt.get("best_acc", 0.0),
            "epoch": ckpt.get("epoch", -1),
        })

    if not results:
        print("No completed experiments found.")
        return

    # Print table
    print(f"\n{'='*80}")
    print(f"{'EXPERIMENT RESULTS':^80}")
    print(f"{'='*80}")
    print(f"{'Experiment':<40} {'Backbone':<15} {'Frac':<8} {'Val Acc':<10}")
    print(f"{'-'*80}")
    for r in sorted(results, key=lambda x: (x["backbone_init"], x["label_fraction"])):
        print(f"{r['experiment']:<40} {r['backbone_init']:<15} {r['label_fraction']:<8.2f} {r['best_val_acc']:<10.4f}")
    print(f"{'='*80}\n")

    # Save CSV
    output_path = PROJECT_ROOT / "reports" / "experiment_results.csv"
    os.makedirs(output_path.parent, exist_ok=True)
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Script Transfer Learning Experiment Runner")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["pretrain", "finetune", "visualize", "collect", "all", "single"],
                        help="Which phase to run")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if checkpoints exist")
    parser.add_argument("--label_fractions", type=float, nargs="+", default=None,
                        help="Custom label fractions (default: 0.01 0.05 0.10 0.25 0.50 1.0)")

    # For --phase single
    parser.add_argument("--backbone_init", type=str, default="random")
    parser.add_argument("--ssl_checkpoint", type=str, default=None)
    parser.add_argument("--label_fraction", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.phase == "pretrain":
        phase_pretrain(args)
    elif args.phase == "finetune":
        phase_finetune(args)
    elif args.phase == "visualize":
        phase_visualize(args)
    elif args.phase == "collect":
        phase_collect_results(args)
    elif args.phase == "all":
        phase_pretrain(args)
        phase_finetune(args)
        phase_visualize(args)
        phase_collect_results(args)
    elif args.phase == "single":
        exp_name = f"{args.backbone_init}_frac{args.label_fraction:.2f}"
        cmd = [
            sys.executable, str(SRC_DIR / "train_hybrid_small.py"),
            "--data_root", str(EKUSH_ROOT),
            "--backbone_init", args.backbone_init,
            "--label_fraction", str(args.label_fraction),
            "--experiment_name", exp_name,
            "--batch_size", "128",
        ]
        if args.ssl_checkpoint:
            cmd.extend(["--ssl_checkpoint", args.ssl_checkpoint])
        run_cmd(cmd, f"Single experiment: {exp_name}")


if __name__ == "__main__":
    main()
