"""FastAPI HTTP front-end for TotalSegmentator."""

from __future__ import annotations

import json
import logging
import os
from importlib import metadata

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from .segment import run_segmentation

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("segmenter")

app = FastAPI(title="dicom-imager-segmenter", version="0.1.0")


def _ts_version() -> str:
    try:
        return metadata.version("TotalSegmentator")
    except Exception:
        return "unknown"


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def root():
    return {
        "service": "dicom-imager-segmenter",
        "totalsegmentator": _ts_version(),
        "tasks": ["total_fast", "total"],
        "endpoints": ["/segment", "/healthz"],
    }


@app.post("/segment")
async def segment(
    nifti: UploadFile = File(...),
    task: str = Form("total_fast"),
    body_seg: bool = Form(False),
):
    if not nifti.filename:
        raise HTTPException(status_code=400, detail="missing nifti upload")

    data = await nifti.read()
    log.info(
        "segment task=%s body_seg=%s bytes=%d filename=%s",
        task, body_seg, len(data), nifti.filename,
    )

    try:
        labels, summary = run_segmentation(data, task=task, body_seg=body_seg)
    except Exception as exc:  # noqa: BLE001 - propagate as 500 with detail
        log.exception("segmentation failed")
        raise HTTPException(status_code=500, detail=f"segmentation failed: {exc}")

    headers = {
        "X-Segmentation-Summary": json.dumps(summary),
        "Content-Disposition": 'attachment; filename="labels.nii.gz"',
    }
    return Response(
        content=labels,
        media_type="application/octet-stream",
        headers=headers,
    )
