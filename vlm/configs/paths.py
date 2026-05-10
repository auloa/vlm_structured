import os
from pathlib import Path

PATH_CONFIG_FILE= Path(__file__).resolve()

VLM_ROOT = PATH_CONFIG_FILE.parent.parent.parent
VLM_MODULE_DIR = VLM_ROOT / VLM_ROOT.name

EXPERIMENTS_DIR = VLM_ROOT / "experiments"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)