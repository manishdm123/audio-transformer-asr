from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.jobs import Job, JobStatus, JobStore, TranscriptionOptions
from app.transcription import transcribe_job

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"

for directory in (UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="audio2text")
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")
templates.env.cache = None
store = JobStore()
executor = ThreadPoolExecutor(max_workers=1)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "jobs": store.list(),
            "active_job": _active_job(),
            "models": ["tiny", "base", "small", "medium", "large-v3"],
            "compute_types": ["int8", "float16", "float32"],
        },
    )


@app.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    model_size: str = Form("small"),
    language: str = Form("en"),
    compute_type: str = Form("int8"),
    word_timestamps: bool = Form(False),
    vad_filter: bool = Form(True),
) -> RedirectResponse:
    job_id = uuid.uuid4().hex
    filename = _safe_filename(audio.filename or "audio")
    upload_path = UPLOAD_DIR / f"{job_id}-{filename}"

    with upload_path.open("wb") as target:
        while chunk := await audio.read(1024 * 1024):
            target.write(chunk)

    options = TranscriptionOptions(
        model_size=model_size,
        language=None if language == "auto" else language,
        compute_type=compute_type,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
    )
    job = Job(
        id=job_id,
        filename=filename,
        upload_path=upload_path,
        output_dir=OUTPUT_DIR,
        options=options,
    )
    store.add(job)
    background_tasks.add_task(lambda: executor.submit(transcribe_job, job_id, store))
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: str) -> HTMLResponse:
    job = _get_job(job_id)
    return templates.TemplateResponse(
        request,
        "job.html",
        {
            "request": request,
            "job": job,
            "jobs": store.list(),
            "models": ["tiny", "base", "small", "medium", "large-v3"],
            "compute_types": ["int8", "float16", "float32"],
        },
    )


@app.get("/jobs/{job_id}/panel", response_class=HTMLResponse)
def job_panel(request: Request, job_id: str) -> HTMLResponse:
    job = _get_job(job_id)
    return templates.TemplateResponse(request, "partials/job_panel.html", {"request": request, "job": job})


@app.get("/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str) -> FileResponse:
    if kind not in {"txt", "md", "srt", "json"}:
        raise HTTPException(status_code=404, detail="Unknown export type")
    job = _get_job(job_id)
    path = OUTPUT_DIR / f"{job.id}.{kind}"
    if job.status != JobStatus.DONE or not path.exists():
        raise HTTPException(status_code=404, detail="Export is not ready")
    return FileResponse(path, filename=f"{Path(job.filename).stem}.{kind}")


def _active_job() -> Job | None:
    for job in store.list():
        if job.status in {JobStatus.RUNNING, JobStatus.QUEUED, JobStatus.DONE}:
            return job
    return None


def _get_job(job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).name).strip("-")
    return cleaned or "audio"


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
