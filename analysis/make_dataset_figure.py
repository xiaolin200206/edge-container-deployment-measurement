"""Build the dataset sample figure from the deposited image classes.

The manuscript's dataset figure needs nothing but the images you already have.
Unzip the class archives from the data deposit, point this script at them, and
it produces a labelled grid at publication resolution.

    python make_dataset_figure.py \
        --background /path/to/background \
        --healthy    /path/to/healthy \
        --disease    /path/to/disease \
        --proxy      /path/to/proxy        # optional, omit if not redistributing
        --rows 3

Sampling is deterministic (fixed seed), so the figure is reproducible, and the
filenames of the chosen images are printed so they can be recorded.
"""
import argparse, random, sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import imread

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import COL2, INK, INK2, use_style, save

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick(folder: Path, n: int, seed: int):
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT)
    if not files:
        raise SystemExit(f"no images found under {folder}")
    rng = random.Random(seed)
    return rng.sample(files, min(n, len(files))), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", required=True)
    ap.add_argument("--healthy", required=True)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--proxy", default=None,
                    help="cross-domain proxy imagery; omit if it cannot be redistributed")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="fig_dataset")
    a = ap.parse_args()

    cols = [("Background", Path(a.background)), ("Healthy", Path(a.healthy)),
            ("Disease (site-collected)", Path(a.disease))]
    if a.proxy:
        cols.append(("Disease (proxy)", Path(a.proxy)))

    use_style()
    fig, axes = plt.subplots(a.rows, len(cols),
                             figsize=(COL2, COL2 * a.rows / len(cols) * 1.06),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.04})
    axes = axes.reshape(a.rows, len(cols))

    for j, (title, folder) in enumerate(cols):
        chosen, total = pick(folder, a.rows, a.seed + j)
        print(f"{title}: {total} images available; showing "
              + ", ".join(p.name for p in chosen))
        for i in range(a.rows):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#d8d7d3"); s.set_linewidth(0.6)
            if i < len(chosen):
                ax.imshow(imread(chosen[i]), aspect="equal")
            if i == 0:
                ax.set_title(title, fontsize=7.2, color=INK, pad=4)

    fig.text(0.5, -0.012,
             "Images sampled deterministically (seed "
             f"{a.seed}) from the deposited class archives.",
             ha="center", fontsize=6.2, color=INK2)
    print("wrote", save(fig, a.out, outdir=str(Path.cwd())))


if __name__ == "__main__":
    main()
