from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASET_DIR = (ROOT / "apple-detection-1").resolve()

DATA_YAML     = DATASET_DIR / "data.yaml"

TRAIN_IMAGES  = DATASET_DIR / "train" / "images"
TRAIN_LABELS  = DATASET_DIR / "train" / "labels"
VAL_IMAGES    = DATASET_DIR / "valid" / "images"
VAL_LABELS    = DATASET_DIR / "valid" / "labels"
TEST_IMAGES   = DATASET_DIR / "test"  / "images"
TEST_LABELS   = DATASET_DIR / "test"  / "labels"

RUNS_DIR = ROOT / "runs"

BASE_WEIGHTS = "yolo11n.pt"

IMGSZ    = 768
EPOCHS   = 100
BATCH    = 16
DEVICE   = "auto"  # auto → multi-GPU if available, else single GPU, else CPU/MPS
WORKERS  = 8
LR0      = 0.01
PATIENCE = 100

# Tracking / line-cross defaults
TRACKER = "bytetrack.yaml"
LINE_POS = 0.5          # horizontal line at 50% image height

# Sanity check dataset paths
def validate_dataset(raise_on_error: bool = True) -> dict:
    checks = {
        "DATASET_DIR": DATASET_DIR.exists(),
        "DATA_YAML": DATA_YAML.exists(),
        "TRAIN_IMAGES": TRAIN_IMAGES.exists(),
        "TRAIN_LABELS": TRAIN_LABELS.exists(),
        "VAL_IMAGES": VAL_IMAGES.exists(),
        "VAL_LABELS": VAL_LABELS.exists(),
        "TEST_IMAGES": TEST_IMAGES.exists(),
        "TEST_LABELS": TEST_LABELS.exists(),
    }
    if raise_on_error and not all(checks.values()):
        missing = [k for k, ok in checks.items() if not ok]
        raise FileNotFoundError(f"Missing required paths: {missing}")
    return checks

def get_data_yaml_path() -> str:
    """String path to pass into Ultralytics train(data=...)."""
    return str(DATA_YAML)
