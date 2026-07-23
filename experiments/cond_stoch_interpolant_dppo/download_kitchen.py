"""Download the official DPPO preprocessed Kitchen dataset folder."""
from __future__ import annotations
import argparse
from pathlib import Path
import gdown

URLS = {
    "kitchen-complete-v0": "https://drive.google.com/drive/u/1/folders/18aqg7KIv-YNXohTsRR7Zmg-RyDtdhkLc",
    "kitchen-partial-v0": "https://drive.google.com/drive/u/1/folders/1zLOx1q4FbJK1ZWLui_vhM2x1fMEkBC2D",
    "kitchen-mixed-v0": "https://drive.google.com/drive/u/1/folders/1HRMM16UC10A00oBqjYOL1E8hS5icwtvo",
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--environment", choices=URLS, default="kitchen-complete-v0")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    files = gdown.download_folder(URLS[a.environment], output=str(a.output), quiet=False)
    if not files:
        raise RuntimeError("official DPPO Kitchen download returned no files")
    print("\n".join(files))
