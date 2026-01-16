"""Visualize nuScenes detection results as a BEV (bird's-eye view) plot for a single frame.

This is a thin, CLI-friendly wrapper around the nuScenes-devkit's
`nuscenes.eval.detection.render.visualize_sample`.

Typical use case in this repo:
- You already have a `results_nusc.json` produced by `tools/test.py` / `dist_test.sh`.
- You want a single BEV image for quick inspection.

Demo (aligned with occ dump folders; recommended):

    python tools/analysis_tools/vis_det_bev_single.py \
        --results test/bev_tiny_det_occ_apollo/Thu_Dec_25_14_36_29_2025/pts_bbox/results_nusc.json \
        --occ_root test/occ_dump_quick --thre 0.25 --scene scene-0916 --frame 0 \
        --dataroot data/nuscenes --version v1.0-mini

Notes:
    - Output will be written to:
            test/occ_dump_quick/thre_0.25/scene-0916/visualization_det_bev/000_bev_det_with_input.png
    - The input image (6 CAM grid) is ON by default. To disable it, add:
            --no_show_input

Notes:
- The BEV rendering is produced by nuScenes-devkit. Colors follow its defaults.
- We generate a stub `EvalBoxes` for GT and convert prediction JSON entries to
  `DetectionBox` for the selected sample.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.render import visualize_sample
from nuscenes.eval.detection.utils import category_to_detection_name


def _postprocess_center_square(
    in_path: str,
    out_path: str,
    *,
    bg_thresh: int = 245,
    pad: int = 10,
    out_size: Optional[int] = None,
) -> None:
    """Crop to foreground and re-pad to a centered square.

    nuScenes-devkit's rendering can include asymmetric margins, making the content
    look visually off-center. We fix this by:
      1) finding the bounding box of non-(near-white) pixels
      2) cropping tightly (with a small pad)
      3) placing it centered on a square white canvas

    Args:
        in_path: Input PNG produced by visualize_sample.
        out_path: Output PNG path.
        bg_thresh: Consider pixels with RGB > bg_thresh as background.
        pad: Extra pixels to keep around the cropped content.
        out_size: If provided, resize the final square canvas to (out_size, out_size).
    """
    from PIL import Image

    im = Image.open(in_path).convert('RGBA')
    a = np.array(im)
    rgb = a[..., :3]
    alpha = a[..., 3]

    # Foreground: non-transparent AND not near-white.
    bg = (rgb > bg_thresh).all(axis=-1)
    fg = (alpha > 0) & (~bg)

    if not np.any(fg):
        # Degenerate: no foreground found. Just copy.
        if os.path.abspath(in_path) != os.path.abspath(out_path):
            Image.open(in_path).save(out_path)
        return

    ys, xs = np.where(fg)
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, im.size[0])
    y1 = min(int(ys.max()) + pad + 1, im.size[1])

    crop = im.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    side = max(cw, ch)

    canvas = Image.new('RGBA', (side, side), (255, 255, 255, 255))
    ox = (side - cw) // 2
    oy = (side - ch) // 2
    canvas.paste(crop, (ox, oy), crop)

    if out_size is not None and out_size > 0 and (side != out_size):
        canvas = canvas.resize((out_size, out_size), resample=Image.BILINEAR)

    canvas.save(out_path)


def _find_sample_token_by_scene_and_frame(
    nusc: NuScenes,
    scene_name: str,
    frame_idx: int,
) -> str:
    """Resolve a nuScenes sample token by (scene name, 0-based frame index).

    The occ dump in this repo uses frame indices like 000, 001, ... per-scene.
    We reproduce the same ordering by traversing the linked-list of samples in a scene:
      scene.first_sample_token -> sample.next -> ...
    """
    if frame_idx < 0:
        raise ValueError(f'frame_idx must be >= 0. Got {frame_idx}')

    # nuScenes stores a list of scene record tokens at nusc.scene.
    scene_token = None
    for sc in nusc.scene:
        if sc.get('name') == scene_name:
            scene_token = sc['token']
            first_sample_token = sc['first_sample_token']
            break
    if scene_token is None:
        raise KeyError(
            f'Cannot find scene name {scene_name!r} in nuScenes {nusc.version}. '
            f'Check that --version matches your dump.'
        )

    token = first_sample_token
    for _ in range(frame_idx):
        rec = nusc.get('sample', token)
        token = rec['next']
        if token == '':
            raise IndexError(
                f'frame_idx={frame_idx} is out of range for scene {scene_name}. '
                f'Scene ended early.'
            )
    return token


def _aligned_output_path(occ_root: str, thre: float, scene: str, frame: int) -> str:
    # Match occ dump layout: <root>/thre_0.25/<scene>/visualization_det_bev/000_bev_det.png
    out_dir = os.path.join(occ_root, f'thre_{thre:.2f}', scene, 'visualization_det_bev')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{frame:03d}_bev_det.png')


def _aligned_input_image_path(occ_root: str, thre: float, scene: str, frame: int) -> str:
    return os.path.join(occ_root, f'thre_{thre:.2f}', scene, 'images', f'{frame:03d}.png')


def _load_input_mosaic_from_nuscenes(
    nusc: NuScenes,
    dataroot: str,
    sample_token: str,
    *,
    layout: str = 'grid',
):
    """Create a camera mosaic for the 6 nuScenes camera views.

    layout:
      - 'grid': 2x3 grid (recommended; clearer when scaled to BEV width)
      - 'row':  1x6 horizontal strip
    """
    import os.path as osp
    import cv2
    import numpy as _np
    from PIL import Image

    sample = nusc.get('sample', sample_token)
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]

    imgs = []

    # Load all 6 cams (fixed order). If a view is missing, insert a placeholder.
    for cam in cams:
        tok = sample['data'].get(cam)
        if not tok:
            imgs.append(None)
            continue

        sd = nusc.get('sample_data', tok)
        p = osp.join(dataroot, sd['filename'])
        if not osp.exists(p):
            imgs.append(None)
            continue
        imgs.append(_np.array(Image.open(p).convert('RGB')))

    if all(im is None for im in imgs):
        return None

    # Target size: choose a consistent height from available images.
    avail = [im for im in imgs if im is not None]
    h = min(im.shape[0] for im in avail)

    # Resize each panel to the same height, keep aspect ratio.
    panels = []
    for cam, im in zip(cams, imgs):
        if im is None:
            # Placeholder width roughly matches typical cam aspect.
            w = int(h * 16 / 9)
            panel = _np.full((h, w, 3), 200, dtype=_np.uint8)
            cv2.putText(
                panel,
                f'{cam} (missing)',
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
            continue

        scale = h / im.shape[0]
        w = max(1, int(im.shape[1] * scale))
        panel = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        # Label each panel to make the 6 views explicit.
        cv2.rectangle(panel, (0, 0), (w, 55), (255, 255, 255), thickness=-1)
        cv2.putText(
            panel,
            cam,
            (15, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)

    if layout == 'row':
        return _np.concatenate(panels, axis=1)

    # 2x3 grid: [0 1 2]
    #          [3 4 5]
    # Make widths uniform per column for clean alignment.
    col_ws = [max(panels[i].shape[1] for i in [0, 3]),
              max(panels[i].shape[1] for i in [1, 4]),
              max(panels[i].shape[1] for i in [2, 5])]

    def pad_to_w(img, w):
        if img.shape[1] == w:
            return img
        pad = _np.full((img.shape[0], w - img.shape[1], 3), 255, dtype=_np.uint8)
        return _np.concatenate([img, pad], axis=1)

    top_row = _np.concatenate([pad_to_w(panels[0], col_ws[0]), pad_to_w(panels[1], col_ws[1]), pad_to_w(panels[2], col_ws[2])], axis=1)
    bot_row = _np.concatenate([pad_to_w(panels[3], col_ws[0]), pad_to_w(panels[4], col_ws[1]), pad_to_w(panels[5], col_ws[2])], axis=1)
    mosaic = _np.concatenate([top_row, bot_row], axis=0)
    return mosaic


def _compose_with_input_image(
    *,
    bev_path: str,
    out_path: str,
    input_rgb,
    top_height: int = 260,
) -> None:
    """Stack input image (top) + BEV (bottom) into one PNG."""
    from PIL import Image

    bev = Image.open(bev_path).convert('RGBA')
    if input_rgb is None:
        bev.save(out_path)
        return

    top = Image.fromarray(input_rgb).convert('RGBA')

    # Make the strip *exactly* (target_w, top_height) so it is always clearly visible.
    target_w = bev.size[0]
    if top_height <= 0:
        top_height = 260

    # Resize keeping aspect ratio so that height == top_height.
    scale = top_height / max(1, top.size[1])
    new_w = max(1, int(top.size[0] * scale))
    top_resized = top.resize((new_w, top_height), resample=Image.BILINEAR)

    if new_w >= target_w:
        # Center-crop to target width.
        x0 = (new_w - target_w) // 2
        top_strip = top_resized.crop((x0, 0, x0 + target_w, top_height))
    else:
        # Pad to target width.
        top_strip = Image.new('RGBA', (target_w, top_height), (255, 255, 255, 255))
        ox = (target_w - new_w) // 2
        top_strip.paste(top_resized, (ox, 0), top_resized)

    gap = 6
    out = Image.new('RGBA', (target_w, top_height + gap + bev.size[1]), (255, 255, 255, 255))
    out.paste(top_strip, (0, 0))
    out.paste(bev, (0, top_height + gap), bev)
    out.save(out_path)


def _build_gt_boxes(nusc: NuScenes, sample_token: str) -> EvalBoxes:
    """Build GT EvalBoxes for a single sample token."""
    gt_box_list: List[DetectionBox] = []

    sample = nusc.get('sample', sample_token)
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        # nuscenes-devkit expects 2D velocity in DetectionBox.
        vel2 = nusc.box_velocity(ann['token'])[:2]
        try:
            det_name = category_to_detection_name(ann['category_name'])
        except Exception:
            det_name = None
        # Skip categories that aren't part of detection eval.
        if not det_name:
            continue

        gt_box_list.append(
            DetectionBox(
                sample_token=ann['sample_token'],
                translation=tuple(ann['translation']),
                size=tuple(ann['size']),
                rotation=tuple(ann['rotation']),
                velocity=tuple(float(x) for x in vel2),
                ego_translation=(0.0, 0.0, 0.0),
                num_pts=int(ann.get('num_pts', -1)),
                detection_name=det_name,
                detection_score=-1.0,
                attribute_name='',
            )
        )

    gt = EvalBoxes()
    gt.add_boxes(sample_token, gt_box_list)
    return gt


def _build_pred_boxes(
    results: Dict[str, Any],
    sample_token: str,
    score_thr: float,
) -> EvalBoxes:
    """Build prediction EvalBoxes for a single sample token from results_nusc.json."""
    pred_box_list: List[DetectionBox] = []

    per_sample: Optional[List[Dict[str, Any]]] = None
    if isinstance(results, dict):
        per_sample = results.get('results', {}).get(sample_token)
    if per_sample is None:
        per_sample = []

    for rec in per_sample:
        if float(rec.get('detection_score', 0.0)) < score_thr:
            continue
        vel = rec.get('velocity', [0.0, 0.0])
        vel2 = (float(vel[0]), float(vel[1])) if len(vel) >= 2 else (0.0, 0.0)
        pred_box_list.append(
            DetectionBox(
                sample_token=rec['sample_token'],
                translation=tuple(rec['translation']),
                size=tuple(rec['size']),
                rotation=tuple(rec['rotation']),
                velocity=vel2,
                ego_translation=tuple(rec.get('ego_translation', (0.0, 0.0, 0.0))),
                num_pts=int(rec.get('num_pts', -1)),
                detection_name=rec['detection_name'],
                detection_score=float(rec.get('detection_score', -1.0)),
                attribute_name=rec.get('attribute_name', ''),
            )
        )

    pred = EvalBoxes()
    pred.add_boxes(sample_token, pred_box_list)
    return pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize nuScenes det results as a BEV plot (single frame).'
    )
    parser.add_argument(
        '--dataroot',
        default='data/nuscenes',
        help='Path to nuScenes dataroot (contains maps/, samples/, sweeps/).',
    )
    parser.add_argument(
        '--version',
        default='v1.0-trainval',
        help='nuScenes version, e.g. v1.0-trainval / v1.0-mini.',
    )
    parser.add_argument(
        '--results',
        required=True,
        help='Path to results_nusc.json produced by evaluation/testing.',
    )

    # Two ways to pick a frame:
    # 1) Directly specify --sample_token.
    # 2) Specify --occ_root/--thre/--scene/--frame to align with occ dump indices.
    parser.add_argument('--sample_token', default='', help='nuScenes sample token to visualize.')
    # Backwards-compat: allow --root as an alias of --occ_root so both scripts
    # can share the same argument names.
    parser.add_argument(
        '--root',
        default='',
        help='Alias of --occ_root (show_dir root used in dist_test --show_dir).',
    )
    parser.add_argument(
        '--occ_root',
        default='',
        help='If provided, enable aligned mode using occ dump root (same as dist_test --show_dir).',
    )
    parser.add_argument(
        '--thre',
        type=float,
        default=0.25,
        help='Occupancy threshold folder name used by occ dump (default: 0.25).',
    )
    parser.add_argument(
        '--scene',
        default='',
        help='Scene name like scene-0916 (required in aligned mode).',
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=-1,
        help='0-based frame index like 0,1,2... (required in aligned mode).',
    )
    parser.add_argument(
        '--score_thr',
        type=float,
        default=0.25,
        help='Only show predicted boxes with score >= this threshold.',
    )
    parser.add_argument(
        '--out',
        default='',
        help=(
            'Output image path. If empty in aligned mode, will write under '
            '<occ_root>/thre_XX.XX/<scene>/visualization_det_bev/{frame}_bev_det.png.'
        ),
    )
    parser.add_argument(
        '--no_gt',
        action='store_true',
        help='If set, don\'t draw GT boxes (pred only).',
    )

    # Image cosmetics.
    parser.add_argument(
        '--center',
        action='store_true',
        help='Post-process output to crop/pad and visually center the BEV plot.',
    )
    parser.add_argument(
        '--center_bg_thresh',
        type=int,
        default=245,
        help='Foreground detection threshold for centering (near-white background).',
    )
    parser.add_argument(
        '--center_pad',
        type=int,
        default=10,
        help='Padding (pixels) kept around cropped content when centering.',
    )
    parser.add_argument(
        '--center_size',
        type=int,
        default=900,
        help='Optional final square size (e.g. 900). 0 means keep crop-derived size.',
    )

    # Default behavior: include the input image (requested by users for quick inspection).
    # Keep an opt-out switch for cases where only BEV is desired.
    parser.add_argument(
        '--no_show_input',
        action='store_true',
        help='If set, do NOT compose input image above the BEV plot.',
    )
    # Backward compatibility: accept the old flag but ignore it (now default).
    # Hidden to avoid confusion in --help.
    parser.add_argument(
        '--show_input',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--input_height',
        type=int,
        default=260,
        help='Height (pixels) of the input-image strip when --show_input is set.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Unify naming with occ visualizer: accept --root as alias of --occ_root.
    if (not args.occ_root) and args.root:
        args.occ_root = args.root

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    results = mmcv.load(args.results)

    if args.occ_root:
        # Clean legacy debug artifacts that may exist from previous iterations.
        # This script should only produce the final visualization image.
        try:
            dbg_dir = os.path.dirname(_aligned_output_path(args.occ_root, args.thre, args.scene, args.frame))
            for fn in os.listdir(dbg_dir):
                if fn.startswith('_debug_') and fn.lower().endswith('.png'):
                    try:
                        os.remove(os.path.join(dbg_dir, fn))
                    except Exception:
                        pass
        except Exception:
            pass
        if not args.scene or args.frame < 0:
            raise ValueError('Aligned mode requires --scene and --frame when --occ_root is set.')
        sample_token = _find_sample_token_by_scene_and_frame(nusc, args.scene, args.frame)

        # In aligned mode, we only keep the composite output (*_with_input.png).
        # So render BEV into a temporary path and delete it after composition.
        aligned_plain = _aligned_output_path(args.occ_root, args.thre, args.scene, args.frame)
        out_path = os.path.splitext(os.path.abspath(aligned_plain))[0] + '__tmp_bev.png'
    else:
        if not args.sample_token:
            raise ValueError('Please provide --sample_token, or use aligned mode with --occ_root/--scene/--frame.')
        sample_token = args.sample_token
        if not args.out:
            raise ValueError('Non-aligned mode requires --out.')
        out_path = args.out

    # In aligned-mode we default to centering for better side-by-side comparison.
    center_enabled = args.center or bool(args.occ_root)

    # Default to show input image (in both aligned and non-aligned modes).
    show_input = not args.no_show_input

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)

    gt = EvalBoxes() if args.no_gt else _build_gt_boxes(nusc, sample_token)
    pred = _build_pred_boxes(results, sample_token, args.score_thr)

    # Important: visualize_sample expects savepath WITHOUT extension; it appends "_bev.png".
    save_prefix = os.path.splitext(os.path.abspath(out_path))[0]
    if save_prefix.endswith('_bev'):
        save_prefix = save_prefix[:-4]

    visualize_sample(nusc, sample_token, gt, pred, savepath=save_prefix + '_bev')

    # `visualize_sample` will save as f"{savepath}.png". We normalize to args.out.
    produced = save_prefix + '_bev.png'

    # Optionally post-process, otherwise just rename/move.
    if center_enabled:
        final_path = out_path
        tmp_path = produced
        if os.path.abspath(produced) != os.path.abspath(out_path):
            # Keep the raw png as a temp file then write centered output to out_path.
            tmp_path = produced
        _postprocess_center_square(
            tmp_path,
            final_path,
            bg_thresh=args.center_bg_thresh,
            pad=args.center_pad,
            out_size=(args.center_size if args.center_size > 0 else None),
        )
        # Cleanup raw file if it differs from the final path.
        if os.path.abspath(tmp_path) != os.path.abspath(final_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        if os.path.abspath(produced) != os.path.abspath(out_path):
            try:
                os.replace(produced, out_path)
            except OSError:
                # Fall back: leave the produced file and ignore rename.
                pass

    # Add a simple legend/annotation for GT vs Pred colors.
    # nuScenes-devkit BEV render uses (by default): GT=green, Pred=blue.
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as _np

        final_path = out_path
        if not os.path.exists(final_path):
            final_path = produced

        im = Image.open(final_path).convert('RGBA')
        draw = ImageDraw.Draw(im)

        # Legend layout.
        margin = 12
        box = 18
        gap = 10

        # Detect the black border/frame area and place legend at its top-right corner.
        arr = _np.array(im.convert('RGB'))
        black = (arr[..., 0] < 40) & (arr[..., 1] < 40) & (arr[..., 2] < 40)
        ys, xs = _np.where(black)
        if len(xs) > 0:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
        else:
            # Fallback: use whole image.
            x0, y0, x1, y1 = 0, 0, im.size[0] - 1, im.size[1] - 1

        # Compute legend top-left (x, y) so legend sits inside frame, top-right.
        # Reserve enough width for text.
        legend_w = 160
        legend_h = box * 2 + gap
        x = max(x0 + margin, x1 - margin - legend_w)
        # Nudge down to avoid overlapping the border line.
        # Empirically, some figures have a slightly thicker/darker top border,
        # so we push the legend a bit further down.
        y = y0 + margin + 22

        # Colors (approximate).
        gt_color = (0, 200, 0, 255)
        pred_color = (0, 120, 255, 255)

        # Font (best-effort).
        font = None
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 16)
        except Exception:
            font = ImageFont.load_default()

        # GT row.
        draw.rectangle([x, y, x + box, y + box], fill=gt_color)
        draw.text((x + box + 10, y - 1), 'GT (green)', fill=(0, 0, 0, 255), font=font)

        # Pred row.
        y2 = y + box + gap
        draw.rectangle([x, y2, x + box, y2 + box], fill=pred_color)
        draw.text((x + box + 10, y2 - 1), 'Pred (blue)', fill=(0, 0, 0, 255), font=font)

        im.save(final_path)
    except Exception:
        # If PIL/font isn't available or anything goes wrong, skip silently.
        pass

    # Optionally compose input image (top) + BEV (bottom).
    if show_input and args.occ_root:
        try:
            from PIL import Image
            import os.path as osp
            import imageio

            input_path = _aligned_input_image_path(args.occ_root, args.thre, args.scene, args.frame)
            input_rgb = None
            if osp.exists(input_path):
                input_rgb = imageio.imread(input_path)
            else:
                input_rgb = _load_input_mosaic_from_nuscenes(nusc, args.dataroot, sample_token, layout='grid')

            # For aligned mode, the desired final output is always:
            # <...>/{frame}_bev_det_with_input.png
            stacked_out = os.path.splitext(_aligned_output_path(args.occ_root, args.thre, args.scene, args.frame))[0] + '_with_input.png'
            _compose_with_input_image(
                bev_path=out_path,
                out_path=stacked_out,
                input_rgb=input_rgb,
                top_height=args.input_height,
            )
            # Delete the temporary BEV image so `{frame}_bev_det.png` is not left behind.
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass

            # Also remove any legacy plain output that might exist from previous runs.
            try:
                legacy_plain = _aligned_output_path(args.occ_root, args.thre, args.scene, args.frame)
                if os.path.exists(legacy_plain):
                    os.remove(legacy_plain)
            except Exception:
                pass
        except Exception:
            pass


if __name__ == '__main__':
    main()
