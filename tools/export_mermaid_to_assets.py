#!/usr/bin/env python3
"""Export Mermaid code blocks in a Markdown file to PNG assets.

Why this exists
--------------
Some Markdown renderers (especially strict or sandboxed previewers) don't support Mermaid.
This script converts ```mermaid code blocks into image files and rewrites the markdown to
reference those images.

Renderer backends
-----------------
- Preferred: Mermaid CLI (mmdc) via Node.js. This produces the most consistent results.
  Requirements:
    - node + npm
    - npm i -g @mermaid-js/mermaid-cli

This repo doesn't vendor Node dependencies by default, so installation is left to the user.
The script will fail with a clear error message if `mmdc` is not found.

Usage
-----
  python3 tools/export_mermaid_to_assets.py \
    --md docs/bevformer_det_occ_multitask_blog.md \
    --out assets/figs/mermaid \
    --prefix bevformer_det_occ \
    --keep-source

It will:
- write images: assets/figs/mermaid/<prefix>_mermaid_<nn>.png
- rewrite markdown:
    - replace each mermaid block with an image reference
    - optionally keep the original mermaid source inside a <details> block

Notes
-----
- This is a deterministic transform (stable numbering per file order).
- If you rerun, it overwrites existing images.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


MERMAID_BLOCK_RE = re.compile(
    r"(^```mermaid\s*\n)(.*?)(^```\s*$)", re.MULTILINE | re.DOTALL
)


@dataclass
class MermaidBlock:
    start: int
    end: int
    code: str


def find_mermaid_blocks(md_text: str) -> List[MermaidBlock]:
    blocks: List[MermaidBlock] = []
    for m in MERMAID_BLOCK_RE.finditer(md_text):
        code = m.group(2)
        blocks.append(MermaidBlock(start=m.start(), end=m.end(), code=code))
    return blocks


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def run_mmdc(input_mmd: Path, output_png: Path, theme: str = "default") -> None:
    mmdc = shutil.which("mmdc")
    if not mmdc:
        raise RuntimeError(
            "`mmdc` (Mermaid CLI) not found.\n"
            "Install it with Node.js: `npm i -g @mermaid-js/mermaid-cli`\n"
            "(You may need to `sudo apt install nodejs npm` first.)"
        )

    # Use a large-ish default width; renderer will scale as needed.
    cmd = [
        mmdc,
        "-i",
        str(input_mmd),
        "-o",
        str(output_png),
        "-t",
        theme,
        "-b",
        "transparent",
        "--scale",
        "2",
    ]
    subprocess.run(cmd, check=True)


def relative_posix_path(from_path: Path, to_path: Path) -> str:
    rel = os.path.relpath(to_path, start=from_path.parent)
    return Path(rel).as_posix()


def build_replacement(
    *,
    md_path: Path,
    out_png_path: Path,
    block_index: int,
    code: str,
    keep_source: bool,
    title: str,
) -> str:
    rel = relative_posix_path(md_path, out_png_path)
    img = f"![{title}](./{rel})" if not rel.startswith("../") else f"![{title}]({rel})"

    if not keep_source:
        return img + "\n"

    fenced = "```mermaid\n" + code.rstrip() + "\n```"
    details = (
        "<details>\n"
        "<summary>Mermaid source</summary>\n\n"
        + fenced
        + "\n\n</details>\n"
    )

    return img + "\n\n" + details


def export(md_path: Path, out_dir: Path, prefix: str, keep_source: bool, theme: str) -> Tuple[int, List[Path]]:
    md_text = md_path.read_text(encoding="utf-8")
    blocks = find_mermaid_blocks(md_text)

    ensure_dir(out_dir)

    # We'll rewrite from back to front to keep indices valid.
    updated = md_text
    written: List[Path] = []

    for i, block in enumerate(reversed(blocks), start=1):
        real_index = len(blocks) - i + 1
        code = block.code

        # Use a stable filename based on order + content hash to avoid accidental collisions
        # if the user inserts new blocks in the middle later.
        fname = f"{prefix}_mermaid_{real_index:02d}_{sha1_short(code)}.png"
        out_png = out_dir / fname

        tmp_mmd = out_dir / f".{prefix}_mermaid_{real_index:02d}.mmd"
        tmp_mmd.write_text(code, encoding="utf-8")

        run_mmdc(tmp_mmd, out_png, theme=theme)
        tmp_mmd.unlink(missing_ok=True)

        written.append(out_png)

        title = f"Mermaid diagram {real_index}"
        repl = build_replacement(
            md_path=md_path,
            out_png_path=out_png,
            block_index=real_index,
            code=code,
            keep_source=keep_source,
            title=title,
        )

        updated = updated[: block.start] + repl + updated[block.end :]

    md_path.write_text(updated, encoding="utf-8")
    written.reverse()
    return len(blocks), written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True, type=Path, help="Markdown file containing ```mermaid blocks")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for generated PNGs")
    ap.add_argument("--prefix", default="diagrams", help="Filename prefix")
    ap.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep Mermaid source in <details> blocks (default is image-only)",
    )
    ap.add_argument("--theme", default="default", help="Mermaid CLI theme (default, dark, neutral, forest)")

    args = ap.parse_args()

    count, written = export(
        md_path=args.md,
        out_dir=args.out,
        prefix=args.prefix,
        keep_source=args.keep_source,
        theme=args.theme,
    )

    print(f"Converted {count} Mermaid blocks")
    for p in written:
        print(f"- {p}")


if __name__ == "__main__":
    main()
