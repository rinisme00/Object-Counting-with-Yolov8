try:
    import comet_ml
except Exception:
    pass

import argparse, os, csv
import torch
from pathlib import Path
from ultralytics import YOLO

# --- our utils
from utils.args import get_train_settings
from utils.Augmentation_and_HyperParameters import get_aug_hyp
from utils.logging import log_to_comet

def auto_device():
    try:
        if torch.cuda.is_available():
            if torch.cuda.device_count() >= 2:
                return [-1, -1] # use 2 most idle GPUs
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def parse_device_arg(dev_str):
    if not dev_str or dev_str == "auto":
        return auto_device()
    s = str(dev_str).strip().lower()
    if s in {"cpu", "mps"}:
        return s
    if s == "all":
        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else "cpu"
    if "," in s:
        return [int(x) for x in s.split(",") if x]
    try:
        return int(s)
    except ValueError:
        return s

def main():
    ap = argparse.ArgumentParser(description="YOLOv8 auto device trainer (multi/single/CPU)")
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="Pretrained model (.pt) or YAML")
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | mps | 0 | 0,1 | all (default: auto)")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs")
    ap.add_argument("--imgsz", type=int, default=None, help="Override image size")
    ap.add_argument("--batch", type=int, default=None, help="Override batch size (int)")
    ap.add_argument("--project", default="runs/detect", help="Project dir (Ultralytics default)")
    ap.add_argument("--name", default="apple_finetune", help="Run name")
    # TensorBoard / ClearML / Comet
    ap.add_argument("--logger", default="none", choices=["none","comet"], help="Post-train logging")
    args = ap.parse_args()

    # Build train kwargs from utils (defaults)
    train_kwargs = get_train_settings()
    aug_kwargs = get_aug_hyp()

    # Minimal overrides from CLI
    if args.epochs is not None: train_kwargs["epochs"] = args.epochs
    if args.imgsz  is not None: train_kwargs["imgsz"]  = args.imgsz
    if args.batch  is not None: train_kwargs["batch"]  = args.batch

    # Device
    train_kwargs["device"]  = parse_device_arg(args.device)
    train_kwargs["project"] = args.project
    train_kwargs["name"]    = args.name

    # Ensure common essentials
    train_kwargs.setdefault("workers", 8)
    train_kwargs.setdefault("patience", 100)
    train_kwargs.setdefault("seed", 0)
    train_kwargs.setdefault("val", True)

    # Merge aug/hyp into train kwargs
    train_kwargs.update(aug_kwargs)

    print(f"[INFO] Using device = {train_kwargs['device']}")
    print(f"[INFO] Model = {args.model}")
    print(f"[INFO] Data  = {args.data}")
    print(f"[INFO] Project/Run = {args.project}/{args.name}")

    # --- Train
    model = YOLO(args.model)
    
    train_kwargs.pop("data", None)
    train_kwargs.pop("model", None)
    
    results = model.train(data=args.data, **train_kwargs)

    # Deduce run directory where results.csv is saved
    run_dir = Path(train_kwargs["project"]) / train_kwargs["name"]
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        alt = Path("runs/detect") / train_kwargs["name"] / "results.csv"
        results_csv = alt if alt.exists() else results_csv

    # --- Logging
    if args.logger == "comet":
        try:
            log_to_comet(results_csv, run_name=args.name, params={"model": args.model, **train_kwargs})
        except Exception as e:
            print(f"[WARN] Comet logging skipped: {e}")

    print(f"[DONE] Best weights: {run_dir / 'weights' / 'best.pt'}")

if __name__ == "__main__":
    main()