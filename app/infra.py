"""Shared infrastructure helpers: Redis connection, RQ queue, S3 client, job state.

Kept deliberately small. Both `app.main` (web) and `app.worker` (RQ worker)
import from here so the wiring lives in one place.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import boto3
import redis
from rq import Queue

# ----------------------------------------------------------------------------
# Redis / RQ

QUEUE_NAME = "segmenter"
JOB_KEY_PREFIX = "jobs:"
WORKER_HEARTBEAT_KEY = "segmenter:worker_last_seen"
WORKER_HEARTBEAT_TTL_S = 60  # /healthz/worker considers worker dead beyond this


def redis_url() -> str:
    """Prefer TLS, fall back to plaintext (for local dev with REDIS_URL=redis://localhost)."""
    return os.environ.get("REDIS_TLS_URL") or os.environ.get("REDIS_URL") or "redis://localhost:6379/0"


_redis: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        url = redis_url()
        # cache-to-go's rediss:// cert isn't in the standard CA bundle on every
        # platform image — disable cert verification only when using TLS. The
        # connection is still encrypted; we just don't pin the cert chain.
        kwargs: dict[str, Any] = {"decode_responses": True}
        if url.startswith("rediss://"):
            kwargs["ssl_cert_reqs"] = None
        _redis = redis.from_url(url, **kwargs)
    return _redis


def get_queue() -> Queue:
    # RQ needs a connection with decode_responses=False for its own serialization,
    # so build a dedicated one rather than reusing get_redis().
    url = redis_url()
    kwargs: dict[str, Any] = {}
    if url.startswith("rediss://"):
        kwargs["ssl_cert_reqs"] = None
    conn = redis.from_url(url, **kwargs)
    return Queue(QUEUE_NAME, connection=conn, default_timeout=3600)


# ----------------------------------------------------------------------------
# S3

S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "rightimaged-production")
S3_PREFIX = "segmenter/jobs"
PRESIGN_TTL_S = 300


_s3 = None


def get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    return _s3


def s3_input_key(job_id: str) -> str:
    return f"{S3_PREFIX}/{job_id}/input.nii.gz"


def s3_labels_key(job_id: str) -> str:
    return f"{S3_PREFIX}/{job_id}/labels.nii.gz"


# ----------------------------------------------------------------------------
# Job state (Redis hash at jobs:{job_id})
#
# Fields are all strings (Redis hash). Complex values (summary, roi_subset) are
# JSON-encoded. Timestamps are unix seconds as strings.

def job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


def _encode(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _decode(field: str, value: str | None) -> Any:
    if value is None:
        return None
    if field in {"summary", "roi_subset"}:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    if field in {"body_seg"}:
        return value == "true"
    if field in {"input_bytes"}:
        try:
            return int(value)
        except ValueError:
            return value
    if field in {"created_at", "started_at", "finished_at"}:
        try:
            return float(value)
        except ValueError:
            return value
    return value


def job_set(job_id: str, **fields: Any) -> None:
    r = get_redis()
    encoded = {k: _encode(v) for k, v in fields.items() if v is not None}
    if not encoded:
        return
    r.hset(job_key(job_id), mapping=encoded)
    # Keep job state for 7 days; ample for polling + debugging.
    r.expire(job_key(job_id), 7 * 24 * 3600)


def job_get(job_id: str) -> dict[str, Any] | None:
    r = get_redis()
    raw = r.hgetall(job_key(job_id))
    if not raw:
        return None
    return {k: _decode(k, v) for k, v in raw.items()}


def worker_heartbeat() -> None:
    r = get_redis()
    r.set(WORKER_HEARTBEAT_KEY, str(time.time()), ex=WORKER_HEARTBEAT_TTL_S)


def worker_last_seen() -> float | None:
    r = get_redis()
    val = r.get(WORKER_HEARTBEAT_KEY)
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None
