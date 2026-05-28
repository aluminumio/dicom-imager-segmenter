"""FastAPI HTTP front-end for the TotalSegmentator worker queue.

Web dyno responsibility: accept uploads, persist them to S3, enqueue an RQ
job on Redis, and serve poll responses + label downloads. All inference runs
on a separate worker dyno (`app.worker`), so this process stays small and
responsive (gunicorn 2x async workers fit comfortably on standard-1x).

Wire contract (unchanged from the in-process version):

    POST /segment           -> 202 {"job_id", "state": "pending"}
    GET  /jobs/{id}         -> {"state": "pending|running|done|error", ...}
    GET  /jobs/{id}/labels  -> labels.nii.gz bytes (streamed from S3)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from importlib import metadata
from typing import Iterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .infra import (
    PRESIGN_TTL_S,
    get_queue,
    get_redis,
    get_s3,
    job_get,
    job_set,
    s3_input_key,
    s3_labels_key,
    worker_last_seen,
    WORKER_HEARTBEAT_TTL_S,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("segmenter.web")

app = FastAPI(title="dicom-imager-segmenter", version="0.3.0")


def _ts_version() -> str:
    try:
        return metadata.version("TotalSegmentator")
    except Exception:
        return "unknown"


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/healthz/worker")
def healthz_worker():
    """Reports whether the RQ worker has heartbeat-ed recently.

    A 503 here means jobs will queue but not run — the worker dyno is dead
    or its heartbeat key has expired. Returns 200 (with a warning) on local
    dev where the worker may not be running.
    """
    last = worker_last_seen()
    now = time.time()
    if last is None:
        return {"ok": False, "last_seen": None, "reason": "no heartbeat key in redis"}
    age = now - last
    healthy = age < WORKER_HEARTBEAT_TTL_S
    return {"ok": healthy, "last_seen": last, "age_s": round(age, 1)}


@app.get("/")
def root():
    return {
        "service": "dicom-imager-segmenter",
        "totalsegmentator": _ts_version(),
        "version": "0.3.0",
        "queue": "redis+rq",
        "tasks": ["total_fast", "total"],
        "endpoints": [
            "POST /segment",
            "GET /jobs/{id}",
            "GET /jobs/{id}/labels",
            "/healthz",
            "/healthz/worker",
        ],
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

    roi_list = [c.strip() for c in roi_subset.split(",") if c.strip()] or None

    data = await nifti.read()
    job_id = uuid.uuid4().hex

    # Upload input to S3 BEFORE enqueueing — if the upload fails, we want the
    # caller to see the failure synchronously rather than discovering it via a
    # mysterious "error" job state later.
    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise HTTPException(status_code=500, detail="AWS_S3_BUCKET not configured")
    try:
        get_s3().put_object(
            Bucket=bucket,
            Key=s3_input_key(job_id),
            Body=data,
            ContentType="application/gzip",
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("s3 input upload failed for job=%s", job_id)
        raise HTTPException(status_code=502, detail=f"s3 upload failed: {exc}") from exc

    job_set(
        job_id,
        state="pending",
        task=task,
        body_seg=body_seg,
        roi_subset=roi_list or [],
        input_bytes=len(data),
        created_at=time.time(),
    )

    # Pass the function by dotted path so the worker imports it fresh — avoids
    # serializing closures, which RQ does support but which makes the queue
    # payload bigger and harder to inspect in `rq info`.
    get_queue().enqueue(
        "app.worker.run_job",
        job_id, task, body_seg, roi_list,
        job_id=job_id,  # RQ's own job id == ours, makes redis introspection easier
        result_ttl=7 * 24 * 3600,
        failure_ttl=7 * 24 * 3600,
    )

    log.info(
        "queue job=%s task=%s body_seg=%s roi_subset=%s bytes=%d",
        job_id, task, body_seg, len(roi_list) if roi_list else 0, len(data),
    )
    return {"job_id": job_id, "state": "pending"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = job_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    out = dict(job)
    out["labels_url"] = f"/jobs/{job_id}/labels" if job.get("state") == "done" else None
    return out


@app.get("/jobs/{job_id}/labels")
def job_labels(job_id: str):
    """Stream labels.nii.gz from S3 through the web dyno.

    We considered 302-redirecting to a presigned S3 URL (cheaper bandwidth-wise),
    but the Rails-side `SegmenterClient` uses `Net::HTTP.start { conn.request {
    res.read_body { ... } } }`, which does NOT follow redirects automatically.
    Streaming through here keeps the wire contract unchanged and adds only
    bandwidth cost — no CPU competition with the worker.
    """
    job = job_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    state = job.get("state")
    if state != "done":
        raise HTTPException(status_code=409, detail=f"job state={state}")

    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise HTTPException(status_code=500, detail="AWS_S3_BUCKET not configured")

    try:
        obj = get_s3().get_object(Bucket=bucket, Key=s3_labels_key(job_id))
    except Exception as exc:  # noqa: BLE001
        log.exception("s3 labels fetch failed for job=%s", job_id)
        raise HTTPException(status_code=502, detail=f"s3 fetch failed: {exc}") from exc

    body = obj["Body"]

    def _iter() -> Iterator[bytes]:
        try:
            for chunk in body.iter_chunks(chunk_size=64 * 1024):
                yield chunk
        finally:
            body.close()

    summary = job.get("summary") or {}
    headers = {
        "X-Segmentation-Summary": json.dumps(summary),
        "Content-Disposition": 'attachment; filename="labels.nii.gz"',
    }
    return StreamingResponse(_iter(), media_type="application/octet-stream", headers=headers)
