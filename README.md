# dicom-imager-segmenter

HTTP wrapper around [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
for the dicom-imager Rails app. Runs in a Python dyno on
[Build.io](https://build.io); Rails posts a NIfTI scan and gets back a labels
NIfTI plus a per-class voxel-count summary.

## Why a separate service

Shipping libtorch + torch-rb inside the Rails dyno OOM-killed bld's build dyno
on the C++ compile of the torch-rb extension. Calling the official Python
package from a sidecar service avoids that entirely — pip pulls pre-built
PyTorch wheels, no in-process libtorch — and the Python TotalSegmentator
distribution gets us the full ensemble + body-part-aware cropping that the
hand-rolled Ruby port lacked.

## API

Segmentation can take minutes on CPU. The platform router enforces a 30 s
response deadline, so `/segment` is async — POST queues a job, then poll
`/jobs/{id}` until `state == "done"` and fetch `/jobs/{id}/labels`.

### `POST /segment` -> 202 `{"job_id", "state": "pending"}`

Multipart form:

| field      | type    | required | default      | notes                          |
|------------|---------|----------|--------------|--------------------------------|
| `nifti`    | file    | yes      | -            | `.nii.gz` scan                 |
| `task`     | string  | no       | `total_fast` | `total`, `total_fast`, ...     |
| `body_seg` | bool    | no       | `false`      | crops to body region first     |

### `GET /jobs/{job_id}` -> job state

```json
{
  "state": "done",
  "task": "total_fast",
  "started_at": 1716901130.4,
  "finished_at": 1716901218.7,
  "summary": {
    "task": "total_fast",
    "shape": [512, 512, 70],
    "spacing_mm": [0.97, 0.97, 5.0],
    "timings": {"load_s": 0.4, "segmentation_s": 84.1, "read_back_s": 0.6},
    "nonzero_counts": [
      {"id": 7, "name": "femur_left", "voxels": 2195205},
      ...
    ]
  },
  "labels_url": "/jobs/{job_id}/labels"
}
```

States: `pending`, `running`, `done`, `error` (with `error` field).

### `GET /jobs/{job_id}/labels` -> labels.nii.gz

`application/octet-stream`. The per-class voxel-count summary is also
echoed in the `X-Segmentation-Summary` response header as JSON.

### `GET /healthz`

Liveness probe.

### `GET /`

Reports loaded TotalSegmentator version and supported tasks.

## Local dev

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Smoke test:

```bash
# 1. Queue the job
JOB=$(curl -sS -X POST http://localhost:8000/segment \
  -F "nifti=@/path/to/scan.nii.gz" \
  -F "task=total_fast" | jq -r .job_id)

# 2. Poll until done
while [ "$(curl -sS http://localhost:8000/jobs/$JOB | jq -r .state)" != "done" ]; do
  sleep 5
done

# 3. Fetch labels
curl -sS http://localhost:8000/jobs/$JOB/labels \
  -o /tmp/labels.nii.gz \
  -D /tmp/labels.headers
```

## Deploy (Build.io)

```bash
bld apps:create dicom-imager-segmenter
bld buildpacks:add -a dicom-imager-segmenter heroku/python
bld ps:scale -a dicom-imager-segmenter web=1:Performance-L
git push https://git.build.io/dicom-imager-segmenter.git master
```

Weights for `total_fast` are pre-fetched at slug compile by
`bin/post_compile`. Other tasks lazy-load on first use.
