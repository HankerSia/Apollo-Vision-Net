#!/usr/bin/env python3
"""Find synchronized NuScenes sensor files for a given sample file.

This repo's `data/nuscenes/v1.0-*/sample.json` appears to be a reduced schema
(without the usual `sample['data']` mapping). So we recover the "same frame"
("same sample") relationship by joining tables:

- sample_data.filename -> sample_data.sample_token
- sample_data.calibrated_sensor_token -> calibrated_sensor.sensor_token
- sensor.channel -> the sensor name (e.g. CAM_FRONT, LIDAR_TOP, RADAR_FRONT)

Given an input filename like:
    samples/CAM_BACK/xxx.jpg
we:
1) Look up the matching `sample_data` row by `filename`.
2) Grab its `sample_token`.
3) Collect *all* sample_data rows with that `sample_token`.
4) Map each row to a channel name via calibrated_sensor + sensor.
5) Print the filenames for all channels in that frame.

Works for images and point clouds as long as the file exists in sample_data.json.

Example:
    python3 tools/nuscenes_find_synced_samples.py \
      --dataroot data/nuscenes --version v1.0-mini \
      --filename samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_channel_mapper(dataroot: Path, version: str):
    base = dataroot / version
    calib_rows = _load_json(base / "calibrated_sensor.json")
    sensor_rows = _load_json(base / "sensor.json")

    calib_by_token: Dict[str, dict] = {r["token"]: r for r in calib_rows}
    sensor_by_token: Dict[str, dict] = {r["token"]: r for r in sensor_rows}

    def channel_of(sample_data_row: dict) -> Optional[str]:
        calib = calib_by_token.get(sample_data_row.get("calibrated_sensor_token"))
        if not calib:
            return None
        s = sensor_by_token.get(calib.get("sensor_token"))
        if not s:
            return None
        return s.get("channel")

    return channel_of


def find_sample_data_by_filename(sample_data_rows: List[dict], filename: str) -> Optional[dict]:
    # Filenames in NuScenes tables are POSIX-like relative paths.
    # We compare as-is; caller should provide relative `samples/...` path.
    for r in sample_data_rows:
        if r.get("filename") == filename:
            return r
    return None


def collect_same_sample(
    sample_data_rows: List[dict],
    sample_token: str,
) -> List[dict]:
    return [r for r in sample_data_rows if r.get("sample_token") == sample_token]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Find other NuScenes sensor files in the same frame.")
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"), help="NuScenes root dir")
    p.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"],
        help="NuScenes metadata version folder under dataroot",
    )
    p.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Relative filename in sample_data.json, e.g. samples/CAM_BACK/xxx.jpg",
    )
    p.add_argument(
        "--check-exists",
        action="store_true",
        help="Also check whether each output file exists under dataroot.",
    )
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional comma-separated channel prefixes to keep, e.g. 'CAM_,LIDAR_,RADAR_'",
    )

    args = p.parse_args(argv)

    base = args.dataroot / args.version
    sample_data_path = base / "sample_data.json"
    if not sample_data_path.exists():
        raise SystemExit(f"Missing: {sample_data_path}")

    sample_data_rows = _load_json(sample_data_path)

    target = find_sample_data_by_filename(sample_data_rows, args.filename)
    if not target:
        raise SystemExit(
            "Could not find filename in sample_data.json. "
            "Make sure you pass a relative path like 'samples/CAM_BACK/xxx.jpg'."
        )

    sample_token = target.get("sample_token")
    sd_token = target.get("token")

    channel_of = _build_channel_mapper(args.dataroot, args.version)

    rows = collect_same_sample(sample_data_rows, sample_token)

    # Build channel -> (filename, token)
    channel_to_files: Dict[str, List[Tuple[str, str]]] = {}
    for r in rows:
        ch = channel_of(r)
        if not ch:
            continue
        channel_to_files.setdefault(ch, []).append((r.get("filename"), r.get("token")))

    prefixes: Tuple[str, ...] = tuple([x for x in (s.strip() for s in args.only.split(",")) if x])

    def keep_channel(ch: str) -> bool:
        if not prefixes:
            return True
        return any(ch.startswith(pref) for pref in prefixes)

    print("Input:")
    print(f"  filename     : {args.filename}")
    print(f"  sample_token : {sample_token}")
    print(f"  sample_data  : {sd_token}")
    print("")
    print("Same-frame channels:")

    for ch in sorted(channel_to_files.keys()):
        if not keep_channel(ch):
            continue
        items = channel_to_files[ch]
        # Normally one file per channel per sample, but we keep the list just in case.
        for fn, tok in items:
            line = f"  {ch}: {fn}  token={tok}"
            if args.check_exists and fn:
                exists = (args.dataroot / fn).exists()
                line += f"  exists={exists}"
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
