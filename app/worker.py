"""RQ worker entrypoint + job function.

Run with: `python -m app.worker`
  (loops forever, sending a heartbeat to Redis between jobs so /healthz/worker
  can report liveness.)

The job function `run_job` is what gets enqueued by the web dyno; it reads the
input NIfTI from S3, runs TotalSegmentator, writes the labels NIfTI back to S3,
and updates the job hash in Redis.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time

from redis import Redis
from rq import Queue, Worker

from .infra import (
    QUEUE_NAME,
    get_s3,
    job_set,
    redis_url,
    s3_input_key,
    s3_labels_key,
    worker_heartbeat,
)
from .segment import run_segmentation

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("segmenter.worker")


# Empirical memory probe: a job that allocates 100 MB chunks of numpy arrays
# until something kills the process. Tells us where the actual wall is, vs the
# user-reported 32 GB available / 46 GB cgroup cap. Trigger by enqueuing:
#   curl -X POST https://segmenter.rightimaged.com/probe_memory
# (the matching endpoint is in app/main.py)
def probe_memory_job(job_id: str, max_gb: int = 8) -> dict:
    import gc
    import resource as _resource
    print(f"PROBE-MEM job={job_id} start max_gb={max_gb}", flush=True)
    def _rss_mb():
        return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024
    import numpy as np
    chunks: list = []
    chunk_size_mb = 100
    target_mb = max_gb * 1024
    try:
        mb = 0
        while mb < target_mb:
            chunks.append(np.ones(chunk_size_mb * 1024 * 1024, dtype=np.uint8))
            mb += chunk_size_mb
            # Touch every page so it's actually resident, not just mmap'd.
            chunks[-1][::4096] = 1
            print(f"PROBE-MEM allocated_MB={mb} ru_maxrss_MB={_rss_mb():.0f}", flush=True)
        print(f"PROBE-MEM completed without death; reached {target_mb} MB", flush=True)
    except MemoryError as e:
        print(f"PROBE-MEM MemoryError at allocated_MB={mb}: {e}", flush=True)
        raise
    finally:
        del chunks
        gc.collect()
    return {"reached_mb": mb}


# RQ job function. Must be importable by the worker (it is — `app.worker.run_job`).
def run_job(job_id: str, task: str, body_seg: bool, roi_subset: list[str] | None) -> dict:
    log.info("job=%s start task=%s body_seg=%s roi_subset=%s", job_id, task, body_seg, roi_subset)
    job_set(job_id, state="running", started_at=time.time())
    worker_heartbeat()

    s3 = get_s3()
    bucket = os.environ["AWS_S3_BUCKET"]
    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_input_key(job_id))
        nifti_bytes = obj["Body"].read()
        log.info("job=%s pulled %d bytes from s3", job_id, len(nifti_bytes))

        labels_bytes, summary = run_segmentation(
            nifti_bytes, task=task, body_seg=body_seg, roi_subset=roi_subset
        )

        s3.put_object(
            Bucket=bucket,
            Key=s3_labels_key(job_id),
            Body=labels_bytes,
            ContentType="application/gzip",
        )
        log.info("job=%s wrote %d label bytes to s3", job_id, len(labels_bytes))

        job_set(
            job_id,
            state="done",
            summary=summary,
            finished_at=time.time(),
            labels_bytes=len(labels_bytes),
        )
        worker_heartbeat()
        return summary
    except BaseException as exc:  # noqa: BLE001 — TS calls sys.exit() on license errors; catch BaseException so we record the failure rather than dying silently with state=running.
        log.exception("job=%s failed", job_id)
        job_set(
            job_id,
            state="error",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=time.time(),
        )
        worker_heartbeat()
        raise


def _make_connection() -> Redis:
    url = redis_url()
    kwargs: dict = {}
    if url.startswith("rediss://"):
        kwargs["ssl_cert_reqs"] = None
    return Redis.from_url(url, **kwargs)


def _log_resource_limits():
    """Print the worker process's actual resource limits at boot.

    `bld ps:exec` runs in a sibling cgroup with its own (small) limits, so it
    can't be used to discover the worker's real cap. We `print()` rather than
    `log.info()` because RQ configures the root logger at import time, which
    causes our subsequent `logging.basicConfig` to no-op and our log.info()
    calls to disappear silently.
    """
    import resource
    print("=== PROBE: worker resource probe ===", flush=True)
    for name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS", "RLIMIT_STACK",
                 "RLIMIT_CPU", "RLIMIT_NOFILE", "RLIMIT_NPROC", "RLIMIT_MEMLOCK",
                 "RLIMIT_CORE"):
        try:
            soft, hard = resource.getrlimit(getattr(resource, name))
            print(f"PROBE rlimit {name}: soft={soft} hard={hard}", flush=True)
        except (ValueError, AttributeError):
            pass
    try:
        with open("/proc/self/cgroup") as f:
            print(f"PROBE /proc/self/cgroup: {f.read().strip()}", flush=True)
    except OSError as e:
        print(f"PROBE cgroup read failed: {e}", flush=True)
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory.high",
                 "/sys/fs/cgroup/memory.current",
                 "/sys/fs/cgroup/memory.swap.max",
                 "/sys/fs/cgroup/cpu.max",
                 "/sys/fs/cgroup/pids.max"):
        try:
            with open(path) as f:
                print(f"PROBE {path} = {f.read().strip()}", flush=True)
        except OSError:
            pass
    try:
        with open("/proc/self/oom_score_adj") as f:
            print(f"PROBE oom_score_adj = {f.read().strip()}", flush=True)
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:", "MemFree:", "SwapTotal:")):
                    print(f"PROBE meminfo: {line.strip()}", flush=True)
    except OSError:
        pass
    print("=== PROBE: end resource probe ===", flush=True)


def main():
    conn = _make_connection()
    queue = Queue(QUEUE_NAME, connection=conn, default_timeout=3600)

    _log_resource_limits()

    # Heartbeat every time the worker loops. RQ doesn't give us a hook for
    # idle-loop callbacks, so we wrap the worker to write a heartbeat on
    # startup; the run_job path also heartbeats while a job runs.
    worker_heartbeat()
    log.info("worker starting, listening on queue=%s", QUEUE_NAME)

    # Forward SIGTERM cleanly so bld restarts/deploys don't leave half-done jobs
    # in a "running" state forever — RQ's death handler will mark the job failed
    # when the worker exits mid-job.
    def _term(signum, frame):
        log.info("worker received signal=%s, exiting", signum)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _term)

    worker = Worker([queue], connection=conn)
    # with_scheduler=False keeps things simple. We don't need delayed jobs.
    worker.work(with_scheduler=False, logging_level=os.environ.get("LOG_LEVEL", "INFO"))


if __name__ == "__main__":
    main()
