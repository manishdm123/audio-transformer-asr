from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class DiarizationTurn:
    start: float
    end: float
    speaker: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TranscriptionOptions:
    model_size: str
    language: str | None
    compute_type: str
    word_timestamps: bool
    vad_filter: bool
    diarization: bool = False
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


@dataclass
class Job:
    id: str
    filename: str
    upload_path: Path
    output_dir: Path
    options: TranscriptionOptions
    status: JobStatus = JobStatus.QUEUED
    stage: str = "Queued"
    error: str | None = None
    progress: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str | None = None
    duration: float | None = None
    segments: list[Segment] = field(default_factory=list)
    diarization_turns: list[DiarizationTurn] = field(default_factory=list)

    @property
    def progress_percent(self) -> int:
        return max(0, min(100, int(self.progress * 100)))


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def update(self, job_id: str, **changes: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc)
            return job
