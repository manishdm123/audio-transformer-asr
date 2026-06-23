from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from app.jobs import Job, JobStatus, JobStore, Segment


def transcribe_job(job_id: str, store: JobStore) -> None:
    job = store.get(job_id)
    if job is None:
        return

    try:
        store.update(job.id, status=JobStatus.RUNNING, stage="Preparing audio", progress=0.05)
        audio_path = _normalized_audio_path(job)

        store.update(job.id, stage="Loading model", progress=0.15)
        from faster_whisper import WhisperModel

        model = WhisperModel(
            job.options.model_size,
            compute_type=job.options.compute_type,
        )

        store.update(job.id, stage="Transcribing", progress=0.25)
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=job.options.language,
            vad_filter=job.options.vad_filter,
            word_timestamps=job.options.word_timestamps,
        )

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        language = getattr(info, "language", None)
        segments: list[Segment] = []

        for segment in segments_iter:
            words = []
            for word in getattr(segment, "words", None) or []:
                words.append(
                    {
                        "start": float(word.start),
                        "end": float(word.end),
                        "word": word.word,
                    }
                )

            item = Segment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
                words=words,
            )
            segments.append(item)

            if duration > 0:
                progress = 0.25 + min(item.end / duration, 1.0) * 0.65
                store.update(job.id, segments=segments.copy(), progress=progress)
            elif len(segments) % 5 == 0:
                store.update(job.id, segments=segments.copy())

        store.update(
            job.id,
            status=JobStatus.DONE,
            stage="Complete",
            progress=1.0,
            language=language,
            duration=duration or None,
            segments=segments,
        )
        write_exports(store.get(job.id) or job)
    except Exception as exc:
        store.update(job.id, status=JobStatus.FAILED, stage="Failed", error=str(exc), progress=1.0)


def write_exports(job: Job) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    base = job.output_dir / job.id
    (base.with_suffix(".txt")).write_text(to_txt(job), encoding="utf-8")
    (base.with_suffix(".md")).write_text(to_markdown(job), encoding="utf-8")
    (base.with_suffix(".srt")).write_text(to_srt(job), encoding="utf-8")
    (base.with_suffix(".json")).write_text(to_json(job), encoding="utf-8")


def to_txt(job: Job) -> str:
    return "\n\n".join(segment.text for segment in job.segments if segment.text).strip() + "\n"


def to_markdown(job: Job) -> str:
    lines = [f"# Transcript: {job.filename}", ""]
    if job.language:
        lines.extend([f"- Language: `{job.language}`", ""])
    for segment in job.segments:
        stamp = _clock(segment.start)
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        lines.append(f"**[{stamp}]** {speaker}{segment.text}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_srt(job: Job) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(job.segments, start=1):
        text = segment.text.strip()
        if not text:
            continue
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_srt_time(segment.start)} --> {_srt_time(segment.end)}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks).strip() + "\n"


def to_json(job: Job) -> str:
    payload = {
        "id": job.id,
        "filename": job.filename,
        "status": job.status.value,
        "language": job.language,
        "duration": job.duration,
        "options": {
            "model_size": job.options.model_size,
            "language": job.options.language,
            "compute_type": job.options.compute_type,
            "word_timestamps": job.options.word_timestamps,
            "vad_filter": job.options.vad_filter,
        },
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "text": segment.text,
                "words": segment.words,
            }
            for segment in job.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _normalized_audio_path(job: Job) -> Path:
    if job.upload_path.stat().st_size == 0:
        raise RuntimeError("The uploaded audio file is empty.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return job.upload_path

    target = job.output_dir / f"{job.id}.wav"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(job.upload_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if target.stat().st_size <= 128:
        raise RuntimeError("Audio normalization produced an empty file. Check that the upload contains readable audio.")
    return target


def _clock(seconds: float) -> str:
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _srt_time(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
