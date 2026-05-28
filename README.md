# dicom-imager-segmenter

HTTP wrapper around [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
for the dicom-imager Rails app. Runs on [Build.io](https://build.io) as a
two-dyno service: a small FastAPI web pod that accepts uploads and serves job
state, plus one or more worker pods that pull jobs off Redis and run inference.

## Why a separate service

Shipping libtorch + torch-rb inside the Rails dyno OOM-killed bld's build dyno
on the C++ compile of the torch-rb extension. Calling the official Python
package from a sidecar service avoids that entirely — pip pulls pre-built
PyTorch wheels, no in-process libtorch — and the Python TotalSegmentator
distribution gets us the full ensemble + body-part-aware cropping that the
hand-rolled Ruby port lacked.

## Architecture

```
  Rails (SegmenterClient)
        |  multipart POST /segment
        v
  +-------------+        enqueue            +----------------+
  | web (FastAPI)| --------------------->   |  Redis (RQ)    |
  |  small dyno  | <---  state HGETALL ---  |  + job state   |
  +-------------+                            +----------------+
        |                                          ^
        |  PUT input                               | dequeue
        v                                          |
  +-------------+                            +----------------+
  |   S3 bucket | <----  GET input  ------   | worker (RQ)    |
  | (rightimaged|  ----  PUT labels  ----->  |  big dyno      |
  |  -production)|                           |  runs TS       |
  +-------------+                            +----------------+
        ^
        |  GET labels  (streamed through web dyno)
        v
   Rails downloads labels.nii.gz
```

- **Web** stays cheap (standard-1x): only serves JSON + streams S3 bytes back.
- **Worker** is sized for inference (tiny = 2 cores / 46 GB; bump if needed).
- **Redis** holds the queue + job state hash. Jobs persist across deploys.
- **S3** holds input + label NIfTIs at `s3://rightimaged-production/segmenter/jobs/{job_id}/`.
- Multiple workers can run concurrently — `bld ps:scale worker=N` and they
  fan out across the queue.

## API

Segmentation can take minutes on CPU. The platform router enforces a 30 s
response deadline, so `/segment` is async — POST queues a job, then poll
`/jobs/{id}` until `state == "done"` and fetch `/jobs/{id}/labels`.

### `POST /segment` -> 202 `{"job_id", "state": "pending"}`

Multipart form:

| field        | type    | required | default      | notes                          |
|--------------|---------|----------|--------------|--------------------------------|
| `nifti`      | file    | yes      | -            | `.nii.gz` scan                 |
| `task`       | string  | no       | `total_fast` | see Tasks below                |
| `body_seg`   | bool    | no       | `false`      | crops to body region first     |
| `roi_subset` | string  | no       | `""`         | comma-sep class names to keep  |

The web dyno uploads the bytes to S3 synchronously, then returns the job_id.
A failed S3 upload returns 502 (you'll know immediately rather than via a
mysterious `error` job state later).

### `GET /jobs/{job_id}` -> job state

```json
{
  "state": "done",
  "task": "total_fast",
  "body_seg": false,
  "roi_subset": [],
  "input_bytes": 88234567,
  "created_at": 1716901130.4,
  "started_at": 1716901131.0,
  "finished_at": 1716901218.7,
  "labels_bytes": 524288,
  "summary": {
    "task": "total_fast",
    "shape": [512, 512, 70],
    "spacing_mm": [0.97, 0.97, 5.0],
    "timings": {"load_s": 0.4, "segmentation_s": 84.1, "read_back_s": 0.6},
    "nonzero_counts": [
      {"id": 7, "name": "femur_left", "voxels": 2195205}
    ]
  },
  "labels_url": "/jobs/{job_id}/labels"
}
```

States: `pending`, `running`, `done`, `error` (with `error` field).

### `GET /jobs/{job_id}/labels` -> labels.nii.gz

`application/octet-stream`, streamed from S3 through the web dyno. The
per-class voxel-count summary is also echoed in the `X-Segmentation-Summary`
response header as JSON.

> **Why not 302-redirect to a presigned S3 URL?** The Rails-side
> `SegmenterClient` uses `Net::HTTP.start { conn.request { res.read_body
> { ... } } }`, which does not follow redirects. Streaming via the web dyno
> keeps the wire contract unchanged; bandwidth is the only cost (no CPU
> contention with the worker).

### `GET /healthz` -> liveness probe (always 200 if FastAPI is up).

### `GET /healthz/worker` -> worker heartbeat

```json
{"ok": true, "last_seen": 1716901218.7, "age_s": 4.2}
```

Returns `ok: false` if the worker hasn't heartbeat-ed within `WORKER_HEARTBEAT_TTL_S`
(60s). Jobs will queue but won't run when this trips.

### Tasks

**Free tasks** (no license required):

| task | modality | classes | notes |
|---|---|---|---|
| `total` | CT | 117 (full body) | Default for general anatomy. Includes humerus_left/right, scapula_left/right, clavicula_left/right at 1.5 mm. |
| `total_fast` | CT | 117 | 3 mm resolution. Quick smoke testing. |
| `total_mr` | MR | 56 | Free body-coverage MR analog of `total`. |
| `body`, `body_mr`, `vertebrae`, `lung_vessels`, ... | varies | varies | See [TS docs](https://github.com/wasserth/TotalSegmentator#subtasks) for the full list. |

**Licensed tasks** (require `TOTALSEG_LICENSE` env var):

| task | modality | what it segments |
|---|---|---|
| `appendicular_bones` | CT | Distal limb bones |
| `tissue_types` | CT | Subcutaneous fat / muscle / bone |
| `tissue_4_types` | CT | Adds visceral fat split |
| `tissue_types_mr` | MR | Same as `tissue_types` on MR |
| `thigh_shoulder_muscles` | CT | Rotator-cuff + thigh muscles |
| `thigh_shoulder_muscles_mr` | MR | Same on MR |
| `vertebrae_body` | CT | Vertebral body sub-segmentation |

License key obtained from <https://backend.totalsegmentator.com/license-academic/>.
Cite [Wasserthal et al., Radiology AI 2023](https://pubs.rsna.org/doi/10.1148/ryai.230024).

## Local dev

```bash
# Terminal 1: redis
redis-server

# Terminal 2: web
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export REDIS_URL=redis://localhost:6379/0
export AWS_S3_BUCKET=rightimaged-production   # or any bucket you own
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
uvicorn app.main:app --reload

# Terminal 3: worker
source .venv/bin/activate
export REDIS_URL=redis://localhost:6379/0
export AWS_S3_BUCKET=rightimaged-production
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
python -m app.worker
```

Smoke test:

```bash
JOB=$(curl -sS -X POST http://localhost:8000/segment \
  -F "nifti=@/path/to/scan.nii.gz" \
  -F "task=total_fast" | jq -r .job_id)

while [ "$(curl -sS http://localhost:8000/jobs/$JOB | jq -r .state)" != "done" ]; do
  sleep 5
done

curl -sS http://localhost:8000/jobs/$JOB/labels \
  -o /tmp/labels.nii.gz \
  -D /tmp/labels.headers
```

## Config vars

| var | required | notes |
|---|---|---|
| `REDIS_URL` or `REDIS_TLS_URL` | yes | RQ + job state. Prefer TLS. |
| `AWS_S3_BUCKET` | yes | e.g. `rightimaged-production` |
| `AWS_ACCESS_KEY_ID` | yes | |
| `AWS_SECRET_ACCESS_KEY` | yes | |
| `AWS_REGION` | yes | e.g. `us-east-1` |
| `TOTALSEG_LICENSE` | for licensed tasks | |
| `TOTALSEG_HOME_DIR` | recommended | e.g. `/workspace/.totalsegmentator` |
| `PIP_NO_CACHE_DIR` | recommended | `1` |
| `LOG_LEVEL` | no | `INFO` by default |

## Deploy (Build.io)

```bash
# One-time config
bld config:set -a dicom-imager-segmenter \
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
  AWS_REGION=us-east-1 AWS_S3_BUCKET=rightimaged-production

# Push
git push https://git.build.io/dicom-imager-segmenter.git master

# Scale: small web, big worker
bld ps:scale -a dicom-imager-segmenter web=1:standard-1x worker=1:tiny
```

Scale workers up for throughput:

```bash
bld ps:scale -a dicom-imager-segmenter worker=3:tiny
```

Or scale a worker up for `total` (1.5mm) latency:

```bash
bld ps:scale -a dicom-imager-segmenter worker=1:performance-2xl
```

Weights for `total_fast` are pre-fetched at slug compile by `bin/post_compile`
when present. Other tasks lazy-load on first use (first request after a
restart pays a one-time download cost).
