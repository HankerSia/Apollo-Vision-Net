#!/usr/bin/env python3
"""Compatibility wrapper for previous filename drift.

The main 3-frame aggregation converter lives at:
  tools/convert_lidar_pcd_sequence_to_occ.py

Some batch scripts/tools may refer to:
  tools/convert_lidar_pcd_seq3_voxelmorph.py

This wrapper forwards all CLI args to the real script.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve()
    # The real converter script in this repo.
    candidates = [
        here.parent / "convert_lidar_pcd_sequence_to_occ.py",
    ]

    real = next((p for p in candidates if p.exists()), None)
    if real is None:
        raise FileNotFoundError(
            "Cannot find the seq3 converter script. Tried: " + ", ".join(str(p) for p in candidates)
        )

    runpy.run_path(str(real), run_name="__main__")


if __name__ == "__main__":
    main()
