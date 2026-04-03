"""
Embedding visualization with t-SNE/UMAP for cross-script transfer analysis.
Extracts embeddings from a trained HybridModel and plots them colored by class.

All figures: no titles, PDF+PNG, journal-quality styling via plot_config.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import (
    DatasetFolder, default_loader, IMG_EXTENSIONS, make_dataset,
)

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

from plot_config import (
    apply_journal_style, save_figure, create_figure, add_grid,
    get_palette, CMAP_SEQUENTIAL, BLUE, GRAY,
)
from model import HybridModel, ResNet50Backbone
from dataset import AlbumentationsImageFolder, get_transforms
from utils import load_resume


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


@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=5000):
    """Extract embeddings (before ArcFace) from model."""
    model.eval()
    embeddings = []
    labels = []
    count = 0

    for imgs, lbls in loader:
        if count >= max_samples:
            break
        imgs = imgs.to(device)
        emb = model(imgs, labels=None)  # (B, d_model)
        embeddings.append(emb.cpu().numpy())
        labels.extend(lbls.numpy().tolist())
        count += imgs.size(0)

    embeddings = np.concatenate(embeddings, axis=0)[:max_samples]
    labels = np.array(labels[:max_samples])
    return embeddings, labels


def plot_tsne(embeddings, labels, class_names, output_path, perplexity=30):
    """Run t-SNE and plot 2D scatter colored by class. No title (journal style)."""
    from sklearn.manifold import TSNE

    print(f"Running t-SNE on {len(embeddings)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeddings)

    num_classes = len(np.unique(labels))

    if num_classes <= 30:
        fig, ax = create_figure(width=7, height=5.5)
        colors = get_palette(num_classes)
        for i in range(num_classes):
            mask = labels == i
            name = class_names[i] if i < len(class_names) else str(i)
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i % len(colors)]],
                       label=name, s=10, alpha=0.65, edgecolors="none")
        ax.legend(fontsize=6, ncol=2, loc="best", markerscale=2,
                  framealpha=0.85, edgecolor="#cccccc")
    else:
        fig, ax = create_figure(width=7, height=5.5)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                             cmap="nipy_spectral", s=3, alpha=0.5, edgecolors="none")
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Class index", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.tick_params(axis="both", which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    save_figure(fig, output_path)


def plot_tsne_comparison(embeddings_dict, labels_dict, class_names, output_path, perplexity=30):
    """Plot t-SNE side-by-side for multiple experiments. No titles."""
    from sklearn.manifold import TSNE

    n_exp = len(embeddings_dict)
    fig, axes = create_figure(width=5 * n_exp, height=4.5, ncols=n_exp)
    if n_exp == 1:
        axes = [axes]

    for idx, (name, emb) in enumerate(embeddings_dict.items()):
        ax = axes[idx]
        lbl = labels_dict[name]

        print(f"Running t-SNE for '{name}' ({len(emb)} samples)...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        coords = tsne.fit_transform(emb)

        num_classes = len(np.unique(lbl))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=lbl,
                             cmap="nipy_spectral", s=3, alpha=0.5, edgecolors="none")
        ax.set_xlabel("t-SNE dim 1")
        if idx == 0:
            ax.set_ylabel("t-SNE dim 2")
        ax.tick_params(axis="both", which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        # Use text annotation instead of title
        ax.text(0.5, -0.08, name.replace("_", " ").title(),
                transform=ax.transAxes, ha="center", fontsize=9, fontstyle="italic")

    fig.subplots_adjust(wspace=0.15)
    save_figure(fig, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Visualization (t-SNE)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root with train/val/test")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max samples for t-SNE (more = slower but better)")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="embeddings")
    return parser.parse_args()


def main():
    args = parse_args()
    apply_journal_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    train_dir = data_root / "train"

    # Class mapping
    class_names = sorted(os.listdir(train_dir))
    all_numeric = all(name.isdigit() for name in class_names)
    if all_numeric:
        class_names = sorted(class_names, key=lambda x: int(x))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = args.num_classes or len(class_to_idx)

    # Load model
    ckpt = load_resume(args.model_path, device)
    resnet_backbone = ResNet50Backbone()
    model = HybridModel(num_classes=num_classes, backbone=resnet_backbone,
                        backbone_channels=2048, input_size=args.input_size)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Data loader
    _, val_tf = get_transforms(args.input_size)
    split_dir = data_root / args.split
    folder = FixedClassImageFolder(str(split_dir), class_to_idx)
    ds = AlbumentationsImageFolder(folder, transform=val_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Extract and plot
    embeddings, labels = extract_embeddings(model, loader, device, max_samples=args.max_samples)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "reports" / "embeddings"
    os.makedirs(output_dir, exist_ok=True)

    plot_tsne(embeddings, labels, class_names,
              output_dir / f"tsne_{args.experiment_name}",
              perplexity=args.perplexity)

    # Save raw embeddings for further analysis
    np.savez(output_dir / f"embeddings_{args.experiment_name}.npz",
             embeddings=embeddings, labels=labels)
    print(f"Raw embeddings saved to: {output_dir / f'embeddings_{args.experiment_name}.npz'}")


if __name__ == "__main__":
    main()
