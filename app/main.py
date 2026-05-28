"""FastAPI HTTP front-end for TotalSegmentator.

Segmentation can take minutes on CPU. The platform router enforces a 30 s
response deadline, so /segment is async:

    POST /segment        -> 202 + {"job_id"}                          (returns immediately)
    GET  /jobs/{id}      -> {"state": "pending|running|done|error", ...}
    GET  /jobs/{id}/labels -> labels.nii.gz                            (when done)

Jobs are kept in memory; restarting the dyno wipes them.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from .segment import run_segmentation

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("segmenter")

app = FastAPI(title="dicom-imager-segmenter", version="0.2.0")

# job_id -> {"state", "summary", "labels_path", "error", "started_at", "finished_at", "task"}
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_DIR = Path(tempfile.gettempdir()) / "segmenter_jobs"
_JOBS_DIR.mkdir(exist_ok=True)


def _ts_version() -> str:
    try:
        return metadata.version("TotalSegmentator")
    except Exception:
        return "unknown"


def _worker(job_id: str, data: bytes, task: str, body_seg: bool, roi_subset: list[str] | None):
    job = _JOBS[job_id]
    job["state"] = "running"
    job["started_at"] = time.time()
    try:
        labels, summary = run_segmentation(data, task=task, body_seg=body_seg, roi_subset=roi_subset)
        labels_path = _JOBS_DIR / f"{job_id}.nii.gz"
        labels_path.write_bytes(labels)
        job["summary"] = summary
        job["labels_path"] = str(labels_path)
        job["state"] = "done"
    except BaseException as exc:  # noqa: BLE001 — TS calls sys.exit() on license errors, which raises SystemExit (not Exception). Catch BaseException so the job is marked as error rather than the worker thread dying silently and leaving "state": "running" forever.
        log.exception("job %s failed", job_id)
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["state"] = "error"
    finally:
        job["finished_at"] = time.time()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def root():
    return {
        "service": "dicom-imager-segmenter",
        "totalsegmentator": _ts_version(),
        "tasks": ["total_fast", "total"],
        "endpoints": ["POST /segment", "GET /jobs/{id}", "GET /jobs/{id}/labels", "/healthz"],
    }


@app.post("/segment", status_code=202)
async def segment(
    nifti: UploadFile = File(...),
    task: str = Form("total_fast"),
    body_seg: bool = Form(False),
    roi_subset: str = Form(""),
):
    if not nifti.filename:
        raise HTTPException(status_code=400, detail="missing nifti upload")

    # Comma-separated class-name list: forces TS to predict only these classes
    # (everything else becomes background). Lets callers anchor a partial-FOV
    # scan (e.g. shoulder-only CT) to the right anatomy so the model can't
    # confuse a humerus for a femur etc.
    roi_list = [c.strip() for c in roi_subset.split(",") if c.strip()] or None

    data = await nifti.read()
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {
        "state": "pending",
        "task": task,
        "body_seg": body_seg,
        "roi_subset": roi_list,
        "input_bytes": len(data),
        "created_at": time.time(),
    }
    log.info("queue job=%s task=%s body_seg=%s roi_subset=%s bytes=%d",
             job_id, task, body_seg, len(roi_list) if roi_list else 0, len(data))

    threading.Thread(
        target=_worker, args=(job_id, data, task, body_seg, roi_list), daemon=True
    ).start()
    return {"job_id": job_id, "state": "pending"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # Hide internal-only fields.
    return {k: v for k, v in job.items() if k != "labels_path"} | {
        "labels_url": f"/jobs/{job_id}/labels" if job.get("state") == "done" else None,
    }


@app.get("/jobs/{job_id}/labels")
def job_labels(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["state"] != "done":
        raise HTTPException(status_code=409, detail=f"job state={job['state']}")
    labels_path = job["labels_path"]
    headers = {
        "X-Segmentation-Summary": json.dumps(job["summary"]),
        "Content-Disposition": 'attachment; filename="labels.nii.gz"',
    }
    return FileResponse(
        labels_path,
        media_type="application/octet-stream",
        headers=headers,
        filename="labels.nii.gz",
    )
