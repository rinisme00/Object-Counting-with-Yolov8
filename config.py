from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"              # where Roboflow export will land (keep default)
RUNS_DIR = ROOT / "runs"                  # Ultralytics default
DEFAULT_DATA_YAML = None                  # set after download, or pass via CLI

# Start from COCO-pretrained (v8 or v11). Swap to yolov11s.pt if you prefer YOLOv11.
BASE_WEIGHTS = "yolov8s.pt"

# Training defaults
IMGSZ = 768
EPOCHS = 100
BATCH = 16
DEVICE = 1
WORKERS = 4
LR0 = 0.005
PATIENCE = 25

# Tracking / line-cross defaults
TRACKER = "bytetrack.yaml"
LINE_POS = 0.5          # horizontal line at 50% image height
