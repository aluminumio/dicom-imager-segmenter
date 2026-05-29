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
    """Log the worker process's actual resource limits as the kernel sees them.

    `bld ps:exec` runs in a sibling cgroup with its own (small) limits, so it
    can't be used to discover the worker's real cap. Print everything here
    once at boot so the log captures what the kernel will enforce against us.
    """
    import resource
    log.info("=== worker resource probe ===")
    # POSIX rlimits the kernel will enforce on this process
    for name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS", "RLIMIT_STACK",
                 "RLIMIT_CPU", "RLIMIT_NOFILE", "RLIMIT_NPROC", "RLIMIT_MEMLOCK",
                 "RLIMIT_CORE"):
        try:
            soft, hard = resource.getrlimit(getattr(resource, name))
            log.info("rlimit %s: soft=%s hard=%s", name, soft, hard)
        except (ValueError, AttributeError):
            pass
    # /proc/self/cgroup tells us our actual cgroup path; with that we can read
    # memory.max from the right place rather than the exec-session's view.
    try:
        with open("/proc/self/cgroup") as f:
            log.info("/proc/self/cgroup: %s", f.read().strip())
    except OSError as e:
        log.info("cgroup read failed: %s", e)
    # Walk the cgroup memory.max file directly. In cgroup v2 unified hierarchy,
    # /sys/fs/cgroup/memory.max is OUR cgroup's max (or "max" for unlimited).
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory.high",
                 "/sys/fs/cgroup/memory.current",
                 "/sys/fs/cgroup/cpu.max"):
        try:
            with open(path) as f:
                log.info("%s = %s", path, f.read().strip())
        except OSError:
            pass
    # OOM score adjustment — if -1000 we're protected, if 0 default, if 1000
    # the kernel will pick us first when something has to die.
    try:
        with open("/proc/self/oom_score_adj") as f:
            log.info("oom_score_adj = %s", f.read().strip())
    except OSError:
        pass
    # Snapshot of available system memory at startup.
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:", "MemFree:", "SwapTotal:")):
                    log.info("meminfo: %s", line.strip())
    except OSError:
        pass
    log.info("=== end resource probe ===")


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
