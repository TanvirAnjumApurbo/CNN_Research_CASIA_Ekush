#!/usr/bin/env python3
"""
CASIA Dataset Rebalancer
========================
Rebalances a CASIA-style dataset (train/val/test splits) so each class has
exactly 210/45/45 images respectively.  Supports dry-run, CLI args, and
multiprocessing for ~1M-image datasets.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found – install with:  pip install tqdm")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (edit these or override via CLI flags)
# ─────────────────────────────────────────────────────────────────────────────
ROOT_PATH   = Path(r"F:\\test_research\data\\CASIA")   # ← dataset root
TRAIN_COUNT = 210
VAL_COUNT   = 45
TEST_COUNT  = 45
RANDOM_SEED = 42
NUM_WORKERS = min(8, (os.cpu_count() or 4))             # parallel processes
LOG_CSV     = Path("rebalance_log.csv")

IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".bmp",
               ".tif", ".tiff", ".webp"}
SPLITS      = ("train", "val", "test")
# ─────────────────────────────────────────────────────────────────────────────


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────── helpers ────────────────────────────

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def safe_dest(src: Path, dest_dir: Path) -> Path:
    """Return a collision-free destination path inside dest_dir."""
    dest = dest_dir / src.name
    if not dest.exists():
        return dest
    stem, suffix = src.stem, src.suffix
    counter = 1
    while True:
        candidate = dest_dir / f"{stem}_c{counter:04d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def make_dup_name(src: Path, idx: int) -> str:
    return f"{src.stem}_dup_{idx:06d}{src.suffix}"


# ─────────────────────────────── per-class worker ───────────────────────────

@dataclass
class ClassResult:
    class_name:      str
    before_total:    int
    after_total:     int
    train_count:     int
    val_count:       int
    test_count:      int
    duplicated:      int
    warning:         str = ""
    error:           str = ""


def process_class(
    class_name: str,
    root: Path,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
    dry_run: bool,
) -> ClassResult:
    """
    Worker function executed in a sub-process for one class.
    Collects all images, shuffles, redistributes across splits.
    """
    target_total = train_n + val_n + test_n
    targets      = {"train": train_n, "val": val_n, "test": test_n}
    warnings     = []

    # ── 1. Collect all images from every split ──────────────────────────────
    all_images: List[Path] = []
    for split in SPLITS:
        split_class_dir = root / split / class_name
        if not split_class_dir.exists():
            warnings.append(f"missing {split}/{class_name}")
            continue
        imgs = [p for p in split_class_dir.iterdir() if p.is_file() and is_image(p)]
        if not imgs:
            warnings.append(f"empty {split}/{class_name}")
        all_images.extend(imgs)

    before_total = len(all_images)
    if before_total == 0:
        return ClassResult(
            class_name=class_name,
            before_total=0, after_total=0,
            train_count=0, val_count=0, test_count=0,
            duplicated=0,
            warning="; ".join(warnings),
            error="no images found",
        )

    # ── 2. Shuffle deterministically ────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(all_images)

    # ── 3. Build assignment list (duplicate only if needed) ─────────────────
    duplicated = 0
    assigned: List[Tuple[Path, str]] = []   # (src_path, split)

    if before_total >= target_total:
        pool = all_images[:target_total]
    else:
        pool = list(all_images)
        needed = target_total - before_total
        extras = rng.choices(all_images, k=needed)
        pool.extend(extras)
        duplicated = needed

    idx = 0
    for split, count in targets.items():
        for _ in range(count):
            assigned.append((pool[idx], split))
            idx += 1

    # ── 4. Execute moves ─────────────────────────────────────────────────────
    # Build a per-split name-count dict to generate unique dup names
    dup_counters: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    counts_after = {"train": 0, "val": 0, "test": 0}

    # Determine which originals need to leave their current home first
    # (to avoid overwriting them before they're re-assigned elsewhere)
    # We use a staging approach: stage → original location is cleared.
    staged: dict[Path, Path] = {}   # original_path → staged_path
    # Only stage originals once (some may appear multiple times as dups)
    originals_used = {src for src, _ in assigned}
    # For originals that remain in their correct split we skip re-moving if
    # position coincides – but safest is to stage all and re-place.

    # Count final destinations per original to detect dups
    from collections import Counter
    src_counts = Counter(src for src, _ in assigned)
    src_dup_idx: dict[Path, int] = {}  # running dup index per src

    errors = []
    if not dry_run:
        # Stage all original files to a temp area inside root
        temp_dir = root / "__rebalance_tmp__" / class_name
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            return ClassResult(
                class_name=class_name,
                before_total=before_total, after_total=0,
                train_count=0, val_count=0, test_count=0,
                duplicated=0,
                error=f"cannot create temp dir: {e}",
            )

        staged_paths: dict[Path, Path] = {}
        for src in originals_used:
            staged_name = src.name
            staged_dest = safe_dest(src, temp_dir)
            try:
                shutil.move(str(src), str(staged_dest))
                staged_paths[src] = staged_dest
            except Exception as e:
                errors.append(f"stage {src.name}: {e}")

    for src, split in assigned:
        dest_dir = root / split / class_name
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        # Resolve actual source (staged or dup from staged)
        if not dry_run:
            staged_src = staged_paths.get(src)
            if staged_src is None:
                errors.append(f"staged path missing for {src.name}")
                continue
        else:
            staged_src = src

        use_count = src_counts[src]
        if use_count > 1:
            # Need copies; we copy from staged (first copy = move, rest = copy)
            dup_i = src_dup_idx.get(src, 0)
            src_dup_idx[src] = dup_i + 1
            if dup_i == 0:
                final_name = src.name
            else:
                final_name = make_dup_name(src, dup_i)
            dest = safe_dest(Path(dest_dir / final_name), dest_dir)
            if not dry_run and staged_src.exists():
                try:
                    if dup_i == 0:
                        shutil.copy2(str(staged_src), str(dest))
                    else:
                        shutil.copy2(str(staged_src), str(dest))
                except Exception as e:
                    errors.append(f"copy dup {src.name}: {e}")
        else:
            dest = safe_dest(staged_src if not dry_run else src, dest_dir)
            if not dry_run and staged_src.exists():
                try:
                    shutil.move(str(staged_src), str(dest))
                except Exception as e:
                    errors.append(f"move {src.name}: {e}")

        counts_after[split] += 1

    # Remove staged originals that were duplicated (already copied out)
    if not dry_run:
        for src, staged_p in staged_paths.items():
            if staged_p.exists():
                try:
                    staged_p.unlink()
                except Exception:
                    pass
        # Clean up temp dir if empty
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    return ClassResult(
        class_name   = class_name,
        before_total = before_total,
        after_total  = sum(counts_after.values()),
        train_count  = counts_after["train"],
        val_count    = counts_after["val"],
        test_count   = counts_after["test"],
        duplicated   = duplicated,
        warning      = "; ".join(warnings),
        error        = "; ".join(errors),
    )


# ─────────────────────────────── orchestrator ───────────────────────────────

def discover_classes(root: Path) -> List[str]:
    """Union of class names across all splits (fast scan, no stat calls)."""
    classes: set[str] = set()
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            log.warning("Split directory not found: %s", split_dir)
            continue
        for entry in split_dir.iterdir():
            if entry.is_dir():
                classes.add(entry.name)
    return sorted(classes)


def write_csv(results: List[ClassResult], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_name", "before_total", "after_total",
            "train_count", "val_count", "test_count",
            "duplicated_count", "warning", "error",
        ])
        for r in results:
            writer.writerow([
                r.class_name, r.before_total, r.after_total,
                r.train_count, r.val_count, r.test_count,
                r.duplicated, r.warning, r.error,
            ])
    log.info("CSV log written → %s", csv_path)


def cleanup_temp(root: Path) -> None:
    tmp = root / "__rebalance_tmp__"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)


def run(
    root:        Path,
    train_n:     int,
    val_n:       int,
    test_n:      int,
    seed:        int,
    num_workers: int,
    dry_run:     bool,
    csv_path:    Path,
) -> None:
    log.info("Root: %s", root)
    log.info("Targets  train=%d  val=%d  test=%d  seed=%d", train_n, val_n, test_n, seed)
    log.info("Workers: %d  |  dry_run=%s", num_workers, dry_run)

    if not root.exists():
        log.error("Root path does not exist: %s", root)
        sys.exit(1)

    cleanup_temp(root)   # remove any leftover staging dir from prior run

    log.info("Discovering classes …")
    classes = discover_classes(root)
    log.info("Found %d classes.", len(classes))

    if not classes:
        log.error("No classes found – check your root path and directory structure.")
        sys.exit(1)

    results: List[ClassResult] = []
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(
                process_class,
                cls, root, train_n, val_n, test_n, seed, dry_run,
            ): cls
            for cls in classes
        }

        with tqdm(total=len(classes), unit="class", desc="Rebalancing") as pbar:
            for future in as_completed(future_map):
                cls = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = ClassResult(
                        class_name=cls,
                        before_total=0, after_total=0,
                        train_count=0, val_count=0, test_count=0,
                        duplicated=0,
                        error=str(exc),
                    )
                results.append(result)
                pbar.set_postfix_str(cls[:30])
                pbar.update(1)

                if result.warning:
                    tqdm.write(f"[WARN]  {cls}: {result.warning}")
                if result.error:
                    tqdm.write(f"[ERROR] {cls}: {result.error}")

    elapsed = time.perf_counter() - t0

    # ── Summary ──────────────────────────────────────────────────────────────
    total_before  = sum(r.before_total  for r in results)
    total_after   = sum(r.after_total   for r in results)
    total_dups    = sum(r.duplicated    for r in results)
    errors_found  = [r for r in results if r.error]
    warnings_found= [r for r in results if r.warning]

    log.info("─" * 60)
    log.info("Done in %.1fs", elapsed)
    log.info("Classes processed : %d", len(results))
    log.info("Images before     : %d", total_before)
    log.info("Images after      : %d", total_after)
    log.info("Duplicates added  : %d", total_dups)
    log.info("Classes w/ errors : %d", len(errors_found))
    log.info("Classes w/ warns  : %d", len(warnings_found))

    results.sort(key=lambda r: r.class_name)
    write_csv(results, csv_path)

    cleanup_temp(root)


# ─────────────────────────────── CLI ────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebalance a CASIA-style dataset across train/val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root",     type=Path, default=ROOT_PATH,   help="Dataset root directory")
    p.add_argument("--train",    type=int,  default=TRAIN_COUNT, help="Target images per class in train")
    p.add_argument("--val",      type=int,  default=VAL_COUNT,   help="Target images per class in val")
    p.add_argument("--test",     type=int,  default=TEST_COUNT,  help="Target images per class in test")
    p.add_argument("--seed",     type=int,  default=RANDOM_SEED, help="Random seed")
    p.add_argument("--workers",  type=int,  default=NUM_WORKERS, help="Parallel worker processes")
    p.add_argument("--log-csv",  type=Path, default=LOG_CSV,     help="Output CSV log path")
    p.add_argument("--dry-run",  action="store_true",            help="Simulate without moving files")
    return p.parse_args()


# ─────────────────────────────── entry point ────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(
        root        = args.root,
        train_n     = args.train,
        val_n       = args.val,
        test_n      = args.test,
        seed        = args.seed,
        num_workers = args.workers,
        dry_run     = args.dry_run,
        csv_path    = args.log_csv,
    )
