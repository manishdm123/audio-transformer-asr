from __future__ import annotations

import hashlib
import mimetypes
import re
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.editing import (
    TranscriptEditError,
    merge_segment,
    rename_speaker,
    replace_text,
    split_segment,
    update_segment_texts,
)
from app.jobs import Job, JobStatus, JobStore, TranscriptionOptions
from app.logging_config import get_logger
from app.transcription import transcribe_job, write_exports

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
STATIC_DIR = BASE_DIR / "app" / "static"

for directory in (UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

app = FastAPI(title="audio2text")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")
templates.env.cache = None
templates.env.filters["clock"] = lambda seconds: _clock(float(seconds))
templates.env.globals["asset_version"] = hashlib.sha256(
    (STATIC_DIR / "app.js").read_bytes() + (STATIC_DIR / "styles.css").read_bytes()
).hexdigest()[:12]
store = JobStore()
executor = ThreadPoolExecutor(max_workers=1)
logger = get_logger("web")


class SegmentTextEdit(BaseModel):
    index: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=100_000)


class SegmentTextEdits(BaseModel):
    edits: list[SegmentTextEdit] = Field(min_length=1, max_length=1_000)


class SpeakerRename(BaseModel):
    current: str = Field(min_length=1, max_length=80)
    replacement: str = Field(min_length=1, max_length=80)


class SegmentSplit(BaseModel):
    character_offset: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=100_000)


class SegmentMerge(BaseModel):
    direction: Literal["previous", "next"]


class TranscriptReplace(BaseModel):
    search: str = Field(min_length=1, max_length=1_000)
    replacement: str = Field(max_length=10_000)
    match_case: bool = False


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    active_job = _active_job()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "jobs": store.list(),
            "active_job": active_job,
            "form_options": active_job.options if active_job else None,
            "models": ["tiny", "base", "small", "medium", "large-v3"],
            "compute_types": ["int8", "float16", "float32"],
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    model_size: str = Form("medium"),
    language: str = Form("en"),
    compute_type: str = Form("int8"),
    word_timestamps: bool = Form(False),
    vad_filter: bool = Form(True),
    diarization: bool = Form(True),
    num_speakers: str = Form(""),
    min_speakers: str = Form(""),
    max_speakers: str = Form(""),
) -> RedirectResponse:
    job_id = uuid.uuid4().hex
    filename = _safe_filename(audio.filename or "audio")
    upload_path = UPLOAD_DIR / f"{job_id}-{filename}"

    with upload_path.open("wb") as target:
        while chunk := await audio.read(1024 * 1024):
            target.write(chunk)

    parsed_num_speakers = _optional_int(num_speakers)
    parsed_min_speakers = _optional_int(min_speakers)
    parsed_max_speakers = _optional_int(max_speakers)
    if parsed_min_speakers and parsed_max_speakers and parsed_min_speakers > parsed_max_speakers:
        raise HTTPException(status_code=400, detail="Minimum speakers cannot be greater than maximum speakers")

    options = TranscriptionOptions(
        model_size=model_size,
        language=None if language == "auto" else language,
        compute_type=compute_type,
        word_timestamps=word_timestamps or diarization,
        vad_filter=vad_filter,
        diarization=diarization,
        num_speakers=parsed_num_speakers,
        min_speakers=parsed_min_speakers,
        max_speakers=parsed_max_speakers,
    )
    job = Job(
        id=job_id,
        filename=filename,
        upload_path=upload_path,
        output_dir=OUTPUT_DIR,
        options=options,
    )
    store.add(job)
    logger.info(
        "job=%s queued file=%s bytes=%d model=%s language=%s compute=%s vad=%s words=%s diarization=%s speakers=%s/%s/%s",
        job.id,
        job.filename,
        upload_path.stat().st_size,
        options.model_size,
        options.language or "auto",
        options.compute_type,
        options.vad_filter,
        options.word_timestamps,
        options.diarization,
        options.num_speakers or "auto",
        options.min_speakers or "auto",
        options.max_speakers or "auto",
    )
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
            "form_options": job.options,
            "models": ["tiny", "base", "small", "medium", "large-v3"],
            "compute_types": ["int8", "float16", "float32"],
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/jobs/{job_id}/panel", response_class=HTMLResponse)
def job_panel(request: Request, job_id: str) -> HTMLResponse:
    job = _get_job(job_id)
    return templates.TemplateResponse(
        request,
        "partials/job_panel.html",
        {"request": request, "job": job},
        headers={"Cache-Control": "no-store"},
    )


@app.get("/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str) -> FileResponse:
    if kind not in {"txt", "md", "srt", "json"}:
        raise HTTPException(status_code=404, detail="Unknown export type")
    job = _get_job(job_id)
    path = job.output_dir / f"{job.id}.{kind}"
    if (job.status != JobStatus.DONE and not job.exports_ready) or not path.exists():
        raise HTTPException(status_code=404, detail="Export is not ready")
    return FileResponse(path, filename=f"{Path(job.filename).stem}.{kind}")


@app.get("/jobs/{job_id}/audio")
def job_audio(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    # Keep the browser-facing resource immutable for the lifetime of the player.
    # The normalized WAV is produced in the background and must never replace an
    # in-flight range request for the original upload.
    path = job.upload_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file is not available")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-store"})


@app.put("/jobs/{job_id}/segments")
def edit_segment_texts(job_id: str, request: SegmentTextEdits) -> dict[str, Any]:
    job, changed = _commit_edit(
        job_id,
        lambda current: update_segment_texts(current, ((edit.index, edit.text) for edit in request.edits)),
    )
    logger.info("job=%s editor saved_segments=%d", job.id, changed)
    return {
        "ok": True,
        "message": f"Saved {changed} changed segment{'s' if changed != 1 else ''}.",
        "segments": len(job.segments),
    }


@app.post("/jobs/{job_id}/speakers/rename")
def edit_speaker_name(job_id: str, request: SpeakerRename) -> dict[str, Any]:
    job, changed = _commit_edit(
        job_id,
        lambda current: rename_speaker(current, request.current, request.replacement),
    )
    logger.info("job=%s editor renamed_speaker=%s affected_segments=%d", job.id, request.current, changed)
    return {
        "ok": True,
        "message": f"Renamed {changed} segment{'s' if changed != 1 else ''}.",
        "segments": len(job.segments),
    }


@app.post("/jobs/{job_id}/segments/{segment_index}/split")
def edit_split_segment(job_id: str, segment_index: int, request: SegmentSplit) -> dict[str, Any]:
    job, new_index = _commit_edit(
        job_id,
        lambda current: split_segment(current, segment_index, request.character_offset, request.text),
    )
    logger.info("job=%s editor split_segment=%d new_segment=%d", job.id, segment_index, new_index)
    return {
        "ok": True,
        "message": "Segment split.",
        "segments": len(job.segments),
        "focus_index": new_index,
    }


@app.post("/jobs/{job_id}/segments/{segment_index}/merge")
def edit_merge_segment(job_id: str, segment_index: int, request: SegmentMerge) -> dict[str, Any]:
    job, merged_index = _commit_edit(
        job_id,
        lambda current: merge_segment(current, segment_index, request.direction),
    )
    logger.info(
        "job=%s editor merged_segment=%d direction=%s result_segment=%d",
        job.id,
        segment_index,
        request.direction,
        merged_index,
    )
    return {
        "ok": True,
        "message": "Segments merged.",
        "segments": len(job.segments),
        "focus_index": merged_index,
    }


@app.post("/jobs/{job_id}/replace")
def edit_replace_text(job_id: str, request: TranscriptReplace) -> dict[str, Any]:
    job, replacements = _commit_edit(
        job_id,
        lambda current: replace_text(current, request.search, request.replacement, request.match_case),
    )
    logger.info("job=%s editor replace_all occurrences=%d match_case=%s", job.id, replacements, request.match_case)
    return {
        "ok": True,
        "message": f"Replaced {replacements} occurrence{'s' if replacements != 1 else ''}.",
        "segments": len(job.segments),
        "replacements": replacements,
    }


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


def _commit_edit(job_id: str, editor: Callable[[Job], Any]) -> tuple[Job, Any]:
    def apply(job: Job) -> Any:
        if job.status != JobStatus.DONE:
            raise TranscriptEditError("The transcript can be edited after the job is complete.")
        result = editor(job)
        write_exports(job)
        return result

    try:
        edited = store.edit(job_id, apply)
    except TranscriptEditError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if edited is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return edited


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).name).strip("-")
    return cleaned or "audio"


def _optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def main() -> None:
    from copy import deepcopy

    import uvicorn

    log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["formatters"]["default"].update(
        fmt="%(asctime)s.%(msecs)03d | %(levelprefix)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_config["formatters"]["access"].update(
        fmt='%(asctime)s.%(msecs)03d | %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, log_config=log_config)


if __name__ == "__main__":
    main()
