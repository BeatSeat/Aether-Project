#!/usr/bin/env python3
"""Download DART model checkpoints and data files.

Usage:
    pip install gdown
    python download_models.py

Large checkpoint files are downloaded via gdown (Google Drive).
SMPL-X body model files require manual download from the official website.
Small files (stand.pkl, config yamls, etc.) are expected to be present
already if you cloned the repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DART_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# File registry — update URLs / file IDs when they become available
# ---------------------------------------------------------------------------
# Each entry: (relative_path, gdown_url_or_id, description, size_hint)
# gdown_url_or_id: a Google Drive file ID or full URL.
#                  Set to None for files that cannot be publicly shared.

CHECKPOINTS: list[tuple[str, str | None, str, str]] = [
    (
        "mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt",
        None,  # TODO: upload and set Google Drive file ID
        "Denoiser checkpoint (primary model for inference)",
        "~400 MB",
    ),
    (
        "mvae/mvae_fps_clip/checkpoint_200000.pt",
        None,  # TODO: upload and set Google Drive file ID
        "Motion VAE checkpoint (auto-encoder for motion primitives)",
        "~150 MB",
    ),
]

# ---------------------------------------------------------------------------
# SMPL-X manual download instructions
# ---------------------------------------------------------------------------

SMPLX_INSTRUCTIONS = """
=============================================================================
  SMPL-X Body Model — Manual Download Required
=============================================================================
The SMPL-X model files are licensed and cannot be redistributed.
Please download them manually:

  1. Visit https://smpl-x.is.tue.mpg.de/
  2. Register / log in and accept the license terms.
  3. Download the "SMPL-X models" package (models_smplx.tgz or similar).
  4. Extract and place the model files under:
       dart/data/smplx_lockedhead_20230207/models_lockedhead/smplx/
     You should see files like:
       - SMPLX_NEUTRAL.npz
       - SMPLX_MALE.npz
       - SMPLX_FEMALE.npz

  Alternatively, for the locked-head version used by DART:
    Download "Locked-head SMPL-X models" from the same site and extract to:
       dart/data/smplx_lockedhead_20230207/

  Note: SMPL-X files are only needed if you run SMPL body model fitting
  or visualization. The core DART inference (server.py) does NOT require
  SMPL-X model files at runtime — it only outputs 22 body joints.
=============================================================================
"""


def file_exists_and_nonempty(path: Path) -> bool:
    """Check if a file exists and has non-zero size."""
    return path.is_file() and path.stat().st_size > 0


def download_with_gdown(url_or_id: str, dest: Path) -> bool:
    """Download a file using gdown. Returns True on success."""
    try:
        import gdown
    except ImportError:
        print("[ERROR] gdown is not installed. Run: pip install gdown")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    # gdown accepts either a full URL or a file ID
    if url_or_id.startswith("http"):
        source = url_or_id
    else:
        source = f"https://drive.google.com/uc?id={url_or_id}"

    print(f"  Downloading from: {source}")
    print(f"  Destination:      {dest}")

    try:
        gdown.download(source, str(dest), quiet=False, fuzzy=True)
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        # Clean up partial download
        if dest.exists():
            dest.unlink()
        return False

    if dest.is_file() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [OK] Downloaded {size_mb:.1f} MB")
        return True
    else:
        print("  [ERROR] Downloaded file is empty or missing.")
        if dest.exists():
            dest.unlink()
        return False


def main() -> None:
    os.chdir(DART_ROOT)
    print(f"DART root: {DART_ROOT}\n")

    # ---- 1. Check small files that should already be present ----
    print("=" * 60)
    print("  Checking small files (should be in repo)")
    print("=" * 60)

    small_files = [
        "data/stand.pkl",
        "data/stand_20fps.pkl",
        "data/action_statistics.json",
        "data/fps_dict.json",
        "data/joint_skin_dist.json",
        "config_files/config_hydra/motion_primitive/mp_h2_f8_r4.yaml",
    ]

    all_small_ok = True
    for rel in small_files:
        p = DART_ROOT / rel
        if file_exists_and_nonempty(p):
            print(f"  [OK] {rel}")
        else:
            print(f"  [MISSING] {rel}")
            all_small_ok = False

    if all_small_ok:
        print("  All small files present.\n")
    else:
        print("  Some small files are missing.")
        print("  Make sure you cloned the repo with LFS or copied data files.\n")

    # ---- 2. Download large checkpoints ----
    print("=" * 60)
    print("  Downloading large checkpoint files")
    print("=" * 60)

    downloaded = 0
    skipped = 0
    manual_needed = []

    for rel_path, url_or_id, desc, size_hint in CHECKPOINTS:
        dest = DART_ROOT / rel_path
        print(f"\n  File: {rel_path}")
        print(f"  Description: {desc}")
        print(f"  Expected size: {size_hint}")

        if file_exists_and_nonempty(dest):
            print("  [SKIP] File already exists.")
            skipped += 1
            continue

        if url_or_id is None:
            print("  [MANUAL] This file is not publicly available for auto-download.")
            manual_needed.append((rel_path, desc, size_hint))
            continue

        if download_with_gdown(url_or_id, dest):
            downloaded += 1
        else:
            print("  [FAIL] Could not download this file.")

    # ---- 3. Summary ----
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exist): {skipped}")

    if manual_needed:
        print(f"\n  The following files need manual download:")
        for rel_path, desc, size_hint in manual_needed:
            print(f"    - {rel_path}")
            print(f"      {desc} ({size_hint})")
            print(f"      Ask the project maintainer for the download link,")
            print(f"      or check the DART paper's supplementary materials.")

    # ---- 4. SMPL-X instructions ----
    print(SMPLX_INSTRUCTIONS)

    # ---- 5. Final check ----
    print("=" * 60)
    print("  Quick verification")
    print("=" * 60)
    all_ok = True
    for rel_path, _, _, _ in CHECKPOINTS:
        dest = DART_ROOT / rel_path
        if file_exists_and_nonempty(dest):
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  [OK] {rel_path} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {rel_path}")
            all_ok = False

    if all_ok:
        print("\n  All checkpoints are in place. You can start the server:")
        print("    cd dart && uvicorn server:app --host 0.0.0.0 --port 8900")
    else:
        print("\n  Some files are still missing. Please download them manually.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
