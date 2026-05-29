"""Wrapper around TotalSegmentator's python_api.

Loads a NIfTI scan, runs the requested task, returns the labels image
plus a summary dict with per-class voxel counts and timings.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


# TS treats `license_number` as a hard requirement: when set, it validates
# against the task's entitlement. Free tasks reject any license_number (even
# a valid one) with "Invalid license number" because they don't have an
# entitlement record. So only forward our key when the task actually needs it.
LICENSED_TASKS = {
    "appendicular_bones",
    "tissue_types",
    "tissue_types_mr",
    "tissue_4_types",
    "vertebrae_body",
    "thigh_shoulder_muscles",
    "thigh_shoulder_muscles_mr",
}



# Lazy import — keeps app importable for /healthz even if torch hasn't
# finished setting up (e.g. during cold start).
_TS = None


def _ts():
    global _TS
    if _TS is None:
        from totalsegmentator.python_api import totalsegmentator as _impl
        _TS = _impl
    return _TS


def _class_map(task: str) -> dict[int, str]:
    """Return {class_id: class_name} for a given task."""
    try:
        from totalsegmentator.map_to_binary import class_map
        return class_map.get(task, {})
    except Exception:
        return {}


def _body_mask_scrub(in_path: Path) -> tuple[int, float]:
    """Zero everything outside a body mask in-place.

    Many shoulder CTs ship with burned-in annotations: white text at the
    corners (patient name, scanner settings), orientation cubes, R/L
    markers. Those pixels carry bone-range HU values, and TotalSegmentator
    treats them as anatomy — both wasting compute and corrupting context
    around the real bones.

    Strategy: compute a per-slice in-plane body mask (HU > -300, the
    connected component containing the image center), OR them together
    across z into a single 2D mask, then close + fill holes. Apply that
    same 2D mask to every slice. A single mask across the volume means
    the boundary doesn't flicker slice-to-slice and the corner regions
    (fixed in image-pixel coords) get killed identically everywhere.
    Returns (kept_voxels, fraction_zeroed) for logging.
    """
    from scipy import ndimage

    img = nib.load(str(in_path))
    rescale_slope = float(img.header.get("scl_slope") or 1.0)
    rescale_inter = float(img.header.get("scl_inter") or 0.0)
    raw = np.asanyarray(img.dataobj).astype(np.float32)
    hu = raw * rescale_slope + rescale_inter

    H, W, D = hu.shape
    cy, cx = H // 2, W // 2
    union = np.zeros((H, W), dtype=bool)
    for z in range(D):
        slc = hu[:, :, z] > -300.0
        if not slc.any():
            continue
        labeled, n = ndimage.label(slc)
        if n == 0:
            continue
        # Body is the component containing the image center. Falling back
        # to the largest component if center isn't tissue (unusual — would
        # happen on a near-empty slice) keeps coverage on the early/late
        # slices where the body cross-section is small.
        center_label = int(labeled[cy, cx])
        if center_label == 0:
            sizes = ndimage.sum(slc, labeled, range(1, n + 1))
            center_label = int(np.argmax(sizes)) + 1
        union |= (labeled == center_label)

    union = ndimage.binary_closing(union, iterations=4)
    union = ndimage.binary_fill_holes(union)
    keep = np.broadcast_to(union[:, :, None], hu.shape)

    # Reset air outside body. Use the original storage dtype: if the data
    # was stored unsigned (e.g. uint16 with intercept), -1024 HU maps to
    # the raw value 0 only if intercept is -1024 exactly — close enough for
    # nnUNet's preprocessing. Otherwise we write the HU-equivalent raw.
    air_raw = (-1024.0 - rescale_inter) / rescale_slope if rescale_slope else 0.0
    out_dtype = img.get_data_dtype()
    if np.issubdtype(out_dtype, np.integer):
        air_raw = np.clip(round(air_raw), np.iinfo(out_dtype).min, np.iinfo(out_dtype).max)
    raw[~keep] = air_raw
    nib.save(nib.Nifti1Image(raw.astype(out_dtype), img.affine, img.header), str(in_path))

    kept = int(keep.sum())
    zeroed_frac = 1.0 - (kept / keep.size)
    return kept, zeroed_frac


def run_segmentation(
    nifti_bytes: bytes,
    task: str = "total_fast",
    body_seg: bool = False,
    roi_subset: list[str] | None = None,
    body_part: str | None = None,
    preprocess: bool = True,
) -> tuple[bytes, dict]:
    """Run TotalSegmentator on a NIfTI scan.

    Returns (labels_nii_gz_bytes, summary_dict).
    Summary contains: task, timings, nonzero_counts (sorted desc), shape, spacing.

    preprocess=True (default) runs `_body_mask_scrub` first to eliminate
    burned-in annotations. body_part is logged but not yet used to constrain
    TS — the model doesn't have a body-part conditioning input; this hint
    is staged for future post-hoc constrained argmax.
    """
    fast = task.endswith("_fast")
    base_task = task[:-5] if fast else task

    with tempfile.TemporaryDirectory(prefix="seg_") as tmp:
        tmp = Path(tmp)
        in_path = tmp / "scan.nii.gz"
        out_path = tmp / "labels.nii.gz"
        in_path.write_bytes(nifti_bytes)

        t0 = time.time()
        img = nib.load(str(in_path))
        load_s = time.time() - t0

        scrub_s = 0.0
        scrub_kept = None
        if preprocess:
            tp = time.time()
            scrub_kept, _ = _body_mask_scrub(in_path)
            scrub_s = round(time.time() - tp, 3)

        t1 = time.time()
        ts_kwargs = dict(
            input=str(in_path),
            output=str(out_path),
            task=base_task,
            fast=fast,
            ml=True,
            body_seg=body_seg,
            quiet=True,
        )
        if base_task in LICENSED_TASKS and os.environ.get("TOTALSEG_LICENSE"):
            ts_kwargs["license_number"] = os.environ["TOTALSEG_LICENSE"]
        if roi_subset:
            ts_kwargs["roi_subset"] = roi_subset
        _ts()(**ts_kwargs)
        seg_s = time.time() - t1

        t2 = time.time()
        labels_img = nib.load(str(out_path))
        labels = np.asanyarray(labels_img.dataobj).astype(np.int32)
        ids, counts = np.unique(labels, return_counts=True)
        cmap = _class_map(base_task)
        nonzero = [
            {
                "id": int(i),
                "name": cmap.get(int(i), f"class_{int(i)}"),
                "voxels": int(c),
            }
            for i, c in zip(ids, counts)
            if int(i) != 0
        ]
        nonzero.sort(key=lambda r: r["voxels"], reverse=True)
        labels_bytes = out_path.read_bytes()
        read_s = time.time() - t2

    summary = {
        "task": task,
        "base_task": base_task,
        "fast": fast,
        "body_seg": body_seg,
        "roi_subset": roi_subset,
        "body_part": body_part,
        "preprocess": preprocess,
        "scrub_kept_voxels": scrub_kept,
        "shape": list(img.shape),
        "spacing_mm": [float(x) for x in img.header.get_zooms()[:3]],
        "timings": {
            "load_s": round(load_s, 3),
            "scrub_s": scrub_s,
            "segmentation_s": round(seg_s, 3),
            "read_back_s": round(read_s, 3),
        },
        "labels_bytes": len(labels_bytes),
        "nonzero_counts": nonzero,
    }
    return labels_bytes, summary
