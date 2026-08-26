from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from typing import Any

from app.jobs import DiarizationTurn, Job, JobStatus, JobStore, Segment
from app.logging_config import get_logger


logger = get_logger("transcription")


@contextmanager
def _operation_heartbeat(job_id: str, operation: str, interval: float = 60.0):
    stop = Event()
    started = time.monotonic()

    def report() -> None:
        while not stop.wait(interval):
            logger.info(
                "job=%s operation=%s still_running elapsed=%.1fs",
                job_id,
                operation,
                time.monotonic() - started,
            )

    reporter = Thread(target=report, name=f"audio2text-{operation}-heartbeat", daemon=True)
    reporter.start()
    try:
        yield
    finally:
        stop.set()
        reporter.join(timeout=0.2)


def transcribe_job(job_id: str, store: JobStore) -> None:
    job = store.get(job_id)
    if job is None:
        logger.warning("job=%s cannot start because it is no longer in the job store", job_id)
        return

    job_started = time.monotonic()
    logger.info(
        "job=%s started file=%s model=%s compute=%s language=%s diarization=%s",
        job.id,
        job.filename,
        job.options.model_size,
        job.options.compute_type,
        job.options.language or "auto",
        job.options.diarization,
    )
    try:
        store.update(job.id, status=JobStatus.RUNNING, stage="Preparing audio", progress=0.05)
        audio_path = _normalized_audio_path(job)
        logger.info(
            "job=%s audio_ready path=%s bytes=%d elapsed=%.1fs",
            job.id,
            audio_path.name,
            audio_path.stat().st_size,
            time.monotonic() - job_started,
        )

        store.update(job.id, stage="Loading model", progress=0.15)
        model_started = time.monotonic()
        logger.info("job=%s loading_whisper_model model=%s", job.id, job.options.model_size)
        from faster_whisper import WhisperModel

        with _operation_heartbeat(job.id, "loading_whisper_model"):
            model = WhisperModel(
                job.options.model_size,
                compute_type=job.options.compute_type,
            )
        logger.info("job=%s whisper_model_loaded elapsed=%.1fs", job.id, time.monotonic() - model_started)

        store.update(job.id, stage="Transcribing", progress=0.25)
        transcription_started = time.monotonic()
        logger.info(
            "job=%s transcription_started vad=%s word_timestamps=%s",
            job.id,
            job.options.vad_filter,
            job.options.word_timestamps,
        )
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=job.options.language,
            vad_filter=job.options.vad_filter,
            word_timestamps=job.options.word_timestamps,
        )

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        language = getattr(info, "language", None)
        segments: list[Segment] = []
        last_progress_log = transcription_started

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

            progress = None
            if duration > 0:
                progress = 0.25 + min(item.end / duration, 1.0) * 0.65
            changes: dict[str, Any] = {"segments": segments.copy()}
            if progress is not None:
                changes["progress"] = progress
            # Publish every segment so the polling UI can show text immediately.
            store.update(job.id, **changes)

            now = time.monotonic()
            if len(segments) == 1 or now - last_progress_log >= 30:
                audio_progress = f"{item.end:.1f}/{duration:.1f}s" if duration > 0 else f"{item.end:.1f}s"
                logger.info(
                    "job=%s transcription_progress segments=%d audio=%s progress=%d%% elapsed=%.1fs",
                    job.id,
                    len(segments),
                    audio_progress,
                    int((progress or 0.25) * 100),
                    now - transcription_started,
                )
                last_progress_log = now

        logger.info(
            "job=%s transcription_complete segments=%d language=%s duration=%.1fs elapsed=%.1fs",
            job.id,
            len(segments),
            language or "unknown",
            duration,
            time.monotonic() - transcription_started,
        )
        store.update(
            job.id,
            language=language,
            duration=duration or None,
            segments=segments.copy(),
            progress=0.91,
        )

        if job.options.diarization:
            store.update(job.id, stage="Writing preliminary exports", progress=0.915)
            write_exports(store.get(job.id) or job)
            store.update(job.id, exports_ready=True)
            logger.info(
                "job=%s preliminary_exports_ready transcript_available_while_diarization_runs=true",
                job.id,
            )

            store.update(job.id, stage="Diarizing speakers (transcript ready)", progress=0.92)
            turns = diarize_audio(audio_path, job)
            segments = assign_speakers(segments, turns)
            store.update(job.id, diarization_turns=turns, segments=segments.copy(), progress=0.98)

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
        store.update(job.id, exports_ready=True)
        logger.info(
            "job=%s complete segments=%d turns=%d total_elapsed=%.1fs",
            job.id,
            len(segments),
            len(job.diarization_turns),
            time.monotonic() - job_started,
        )
    except Exception as exc:
        failed_stage = job.stage
        error_message = str(exc) or type(exc).__name__
        store.update(job.id, status=JobStatus.FAILED, stage="Failed", error=error_message, progress=1.0)
        logger.exception(
            "job=%s failed stage=%s elapsed=%.1fs error=%s",
            job.id,
            failed_stage,
            time.monotonic() - job_started,
            error_message,
        )


def write_exports(job: Job) -> None:
    export_started = time.monotonic()
    job.output_dir.mkdir(parents=True, exist_ok=True)
    base = job.output_dir / job.id
    exports = {
        ".txt": to_txt(job),
        ".md": to_markdown(job),
        ".srt": to_srt(job),
        ".json": to_json(job),
    }
    temporary_paths = []
    for suffix, content in exports.items():
        target = base.with_suffix(suffix)
        temporary = target.with_suffix(f"{suffix}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary_paths.append((temporary, target))
    for temporary, target in temporary_paths:
        temporary.replace(target)
    logger.info(
        "job=%s exports_written formats=txt,md,srt,json output=%s elapsed=%.2fs",
        job.id,
        job.output_dir,
        time.monotonic() - export_started,
    )


def to_txt(job: Job) -> str:
    lines = []
    last_speaker = None
    for segment in job.segments:
        text = segment.text.strip()
        if not text:
            continue
        speaker = segment.speaker
        if speaker and speaker != last_speaker:
            lines.append(f"{speaker}:")
            last_speaker = speaker
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


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
        if segment.speaker:
            text = f"{segment.speaker}: {text}"
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
            "diarization": job.options.diarization,
            "num_speakers": job.options.num_speakers,
            "min_speakers": job.options.min_speakers,
            "max_speakers": job.options.max_speakers,
        },
        "diarization_turns": [
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": turn.speaker,
            }
            for turn in job.diarization_turns
        ],
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


def diarize_audio(audio_path: Path, job: Job) -> list[DiarizationTurn]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "Speaker diarization requires HF_TOKEN. Accept the pyannote model terms on Hugging Face, "
            "then run the app with HF_TOKEN set."
        )

    from pyannote.audio import Pipeline

    pipeline_started = time.monotonic()
    logger.info("job=%s loading_diarization_pipeline model=community-1", job.id)
    with _operation_heartbeat(job.id, "loading_diarization_pipeline"):
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    if pipeline is None:
        raise RuntimeError(
            "Could not load pyannote/speaker-diarization-community-1. Confirm your HF_TOKEN has access "
            "and that you accepted the model terms on Hugging Face."
        )
    kwargs = _speaker_count_kwargs(job)
    logger.info(
        "job=%s diarization_pipeline_loaded elapsed=%.1fs speaker_constraints=%s",
        job.id,
        time.monotonic() - pipeline_started,
        kwargs or "auto",
    )
    diarization_started = time.monotonic()
    logger.info("job=%s diarization_started audio=%s", job.id, audio_path.name)
    with _operation_heartbeat(job.id, "diarization"):
        output = pipeline(str(audio_path), **kwargs)
    diarization = getattr(output, "speaker_diarization", output)
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    if exclusive is not None:
        diarization = exclusive

    turns = []
    if hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(DiarizationTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker)))
    else:
        for item in diarization:
            parsed = _parse_diarization_item(item)
            if parsed is not None:
                turns.append(parsed)

    result = sorted(turns, key=lambda turn: (turn.start, turn.end, turn.speaker))
    logger.info(
        "job=%s diarization_complete turns=%d speakers=%d elapsed=%.1fs",
        job.id,
        len(result),
        len({turn.speaker for turn in result}),
        time.monotonic() - diarization_started,
    )
    return result


def assign_speakers(segments: list[Segment], turns: list[DiarizationTurn]) -> list[Segment]:
    if not turns:
        return segments

    diarized: list[Segment] = []
    for segment in segments:
        if segment.words:
            diarized.extend(_split_segment_by_words(segment, turns))
        else:
            segment.speaker = _speaker_for_span(segment.start, segment.end, turns)
            diarized.append(segment)
    return _merge_neighboring_segments(diarized)


def _split_segment_by_words(segment: Segment, turns: list[DiarizationTurn]) -> list[Segment]:
    chunks: list[Segment] = []
    current_speaker = None
    current_words: list[dict[str, Any]] = []

    for word in segment.words:
        start = float(word.get("start", segment.start))
        end = float(word.get("end", start))
        speaker = _speaker_for_span(start, end, turns)
        if current_words and speaker != current_speaker:
            chunks.append(_segment_from_words(current_words, current_speaker))
            current_words = []
        current_speaker = speaker
        current_words.append(word)

    if current_words:
        chunks.append(_segment_from_words(current_words, current_speaker))

    if not chunks:
        segment.speaker = _speaker_for_span(segment.start, segment.end, turns)
        return [segment]
    return chunks


def _segment_from_words(words: list[dict[str, Any]], speaker: str | None) -> Segment:
    text = "".join(str(word.get("word", "")) for word in words).strip()
    return Segment(
        start=float(words[0].get("start", 0.0)),
        end=float(words[-1].get("end", words[0].get("start", 0.0))),
        text=text,
        speaker=speaker,
        words=words.copy(),
    )


def _merge_neighboring_segments(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if not segment.text.strip():
            continue
        previous = merged[-1] if merged else None
        if previous and previous.speaker == segment.speaker and segment.start - previous.end < 1.0:
            previous.end = segment.end
            previous.text = f"{previous.text.rstrip()} {segment.text.lstrip()}".strip()
            previous.words.extend(segment.words)
        else:
            merged.append(segment)
    return merged


def _speaker_for_span(start: float, end: float, turns: list[DiarizationTurn]) -> str | None:
    midpoint = start + max(end - start, 0.0) / 2
    best_speaker = None
    best_overlap = 0.0

    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
        if turn.start <= midpoint <= turn.end and best_overlap == 0.0:
            best_speaker = turn.speaker

    return best_speaker


def _speaker_count_kwargs(job: Job) -> dict[str, int]:
    kwargs = {}
    if job.options.num_speakers:
        kwargs["num_speakers"] = job.options.num_speakers
    else:
        if job.options.min_speakers:
            kwargs["min_speakers"] = job.options.min_speakers
        if job.options.max_speakers:
            kwargs["max_speakers"] = job.options.max_speakers
    return kwargs


def _parse_diarization_item(item: Any) -> DiarizationTurn | None:
    if not isinstance(item, tuple) or len(item) < 2:
        return None
    turn = item[0]
    speaker = item[-1]
    if not hasattr(turn, "start") or not hasattr(turn, "end"):
        return None
    return DiarizationTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker))


def _normalized_audio_path(job: Job) -> Path:
    if job.upload_path.stat().st_size == 0:
        raise RuntimeError("The uploaded audio file is empty.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("job=%s ffmpeg_not_found using_original_upload=%s", job.id, job.upload_path.name)
        return job.upload_path

    job.output_dir.mkdir(parents=True, exist_ok=True)
    target = job.output_dir / f"{job.id}.wav"
    temporary = job.output_dir / f".{job.id}.normalizing.wav"
    normalization_started = time.monotonic()
    logger.info("job=%s ffmpeg_normalization_started binary=%s", job.id, ffmpeg)
    temporary.unlink(missing_ok=True)
    try:
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
                str(temporary),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if temporary.stat().st_size <= 128:
            raise RuntimeError(
                "Audio normalization produced an empty file. Check that the upload contains readable audio."
            )
        temporary.replace(target)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or "").strip().splitlines()
        tail = " | ".join(details[-4:]) if details else str(exc)
        raise RuntimeError(f"Audio normalization failed: {tail}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    logger.info(
        "job=%s ffmpeg_normalization_complete output_bytes=%d elapsed=%.1fs",
        job.id,
        target.stat().st_size,
        time.monotonic() - normalization_started,
    )
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
    total_millis = max(0, int(round(seconds * 1000)))
    total, millis = divmod(total_millis, 1000)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
