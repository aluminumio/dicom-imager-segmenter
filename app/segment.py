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


def run_segmentation(
    nifti_bytes: bytes,
    task: str = "total_fast",
    body_seg: bool = False,
    roi_subset: list[str] | None = None,
) -> tuple[bytes, dict]:
    """Run TotalSegmentator on a NIfTI scan.

    Returns (labels_nii_gz_bytes, summary_dict).
    Summary contains: task, timings, nonzero_counts (sorted desc), shape, spacing.
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

        t1 = time.time()
        # ml=True returns a single multilabel NIfTI rather than per-class masks.
        # license_number is passed to TS when set so licensed tasks (e.g.
        # thigh_shoulder_muscles, tissue_types) can download their weights.
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
        "shape": list(img.shape),
        "spacing_mm": [float(x) for x in img.header.get_zooms()[:3]],
        "timings": {
            "load_s": round(load_s, 3),
            "segmentation_s": round(seg_s, 3),
            "read_back_s": round(read_s, 3),
        },
        "labels_bytes": len(labels_bytes),
        "nonzero_counts": nonzero,
    }
    return labels_bytes, summary
