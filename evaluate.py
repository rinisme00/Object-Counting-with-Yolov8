import argparse
from pathlib import Path

# Use a non-GUI backend for headless servers BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

def safe_plot(df, xcol, ycols, title, ylabel, out_path):
    present = [c for c in ycols if c in df.columns]
    if not present:
        print(f"[SKIP] None of {ycols} found in CSV for '{title}'")
        return
    fig = plt.figure()
    for c in present:
        plt.plot(df[xcol], df[c], label=c)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Ultralytics results.csv")
    ap.add_argument("--out", default="plots", help="Output folder for PNGs")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"[ERROR] CSV not found: {csv_path}")

    # Read CSV (fallback to python engine if needed)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, engine="python")

    # Ensure epoch column
    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(len(df)))

    # Common Ultralytics columns
    train_losses = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    val_losses   = ["val/box_loss",   "val/cls_loss",   "val/dfl_loss"]
    metrics      = ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                    "metrics/precision(B)", "metrics/recall(B)"]
    lrs          = ["lr/pg0", "lr/pg1", "lr/pg2"]

    safe_plot(df, "epoch", train_losses,
              "Training Losses (box/cls/dfl)", "loss", out_dir / "train_losses.png")
    safe_plot(df, "epoch", val_losses,
              "Validation Losses (box/cls/dfl)", "loss", out_dir / "val_losses.png")
    safe_plot(df, "epoch", metrics,
              "Validation Metrics (mAP/Prec/Rec)", "score", out_dir / "val_metrics.png")
    safe_plot(df, "epoch", lrs,
              "Learning Rates (param groups)", "lr", out_dir / "learning_rates.png")

    print(f"[DONE] Plots saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()