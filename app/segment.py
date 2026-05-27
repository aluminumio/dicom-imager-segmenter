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
        _ts()(
            input=str(in_path),
            output=str(out_path),
            task=base_task,
            fast=fast,
            ml=True,
            body_seg=body_seg,
            quiet=True,
        )
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
