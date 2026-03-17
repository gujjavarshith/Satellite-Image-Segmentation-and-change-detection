from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent

# Prefer existing processed path if one already exists.
_candidate_a = BASE_DIR / "utils" / "data" / "processed"
_candidate_b = BASE_DIR / "data" / "processed"
if _candidate_a.exists():
	PROCESSED_DIR = _candidate_a
else:
	PROCESSED_DIR = _candidate_b

RAW_DIR = BASE_DIR / "data" / "raw"

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
NUM_CLASSES = 7
CLASS_NAMES = [
	"non_change",
	"low_vegetation",
	"nvg_surface",
	"tree",
	"water",
	"building",
	"playground",
]

# Raw grayscale values present in processed label PNGs.
# These are mapped in-order to class ids 0..NUM_CLASSES-1.
LABEL_VALUES = [0, 29, 38, 75, 76, 128, 149]

# OpenCV writes/reads in BGR order.
CLASS_COLORS_BGR = [
	(0, 0, 0),      # non_change      black
	(34, 139, 34),  # low_vegetation  green
	(80, 127, 255), # nvg_surface     coral
	(0, 100, 0),    # tree			  dark green
	(255, 0, 0),    # water			  blue
	(0, 0, 255),    # building        red
	(0, 255, 255),  # playground      cyan/yellow
]

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4

SEED = 42
NUM_WORKERS = 2
PIN_MEMORY = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = BASE_DIR / "runs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = SAVE_DIR / "segformer_change_best.pt"


def get_split_dir(split_name: str) -> Path:
	return PROCESSED_DIR / split_name


def get_image_dir(split_name: str) -> Path:
	return get_split_dir(split_name) / "im1"


def get_label_dir(split_name: str) -> Path:
	return get_split_dir(split_name) / "labels"
