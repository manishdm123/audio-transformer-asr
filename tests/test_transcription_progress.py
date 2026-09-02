from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.jobs import DiarizationTurn, Job, JobStatus, JobStore, TranscriptionOptions
from app.transcription import _normalized_audio_path, transcribe_job


class TranscriptionProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.upload = self.directory / "recording.wav"
        self.upload.write_bytes(b"RIFF-test-audio")
        self.store = JobStore()

    def make_job(self, *, diarization: bool = False) -> Job:
        job = Job(
            id="progress-job",
            filename="recording.wav",
            upload_path=self.upload,
            output_dir=self.directory,
            options=TranscriptionOptions("tiny", "en", "int8", diarization, True, diarization=diarization),
        )
        self.store.add(job)
        return job

    def test_each_segment_is_published_even_when_duration_is_unknown(self) -> None:
        job = self.make_job()
        observed_segment_counts: list[int] = []

        class FakeWhisperModel:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def transcribe(self, *_args, **_kwargs):
                def segments():
                    yield SimpleNamespace(start=0.0, end=2.0, text=" First chunk", words=[])
                    observed_segment_counts.append(len(self_store.get(job.id).segments))
                    yield SimpleNamespace(start=2.0, end=4.0, text=" Second chunk", words=[])

                return segments(), SimpleNamespace(duration=0.0, language="en")

        self_store = self.store
        fake_module = SimpleNamespace(WhisperModel=FakeWhisperModel)
        with (
            patch.dict(sys.modules, {"faster_whisper": fake_module}),
            patch("app.transcription._normalized_audio_path", return_value=self.upload),
        ):
            transcribe_job(job.id, self.store)

        self.assertEqual(observed_segment_counts, [1])
        self.assertEqual([segment.text for segment in job.segments], ["First chunk", "Second chunk"])
        self.assertEqual(job.status, JobStatus.DONE)
        self.assertTrue(job.exports_ready)

    def test_stage_timestamp_changes_only_when_the_stage_changes(self) -> None:
        job = self.make_job()
        original = job.stage_started_at - timedelta(seconds=5)
        job.stage_started_at = original

        self.store.update(job.id, progress=0.1, stage="Queued")
        self.assertEqual(job.stage_started_at, original)

        self.store.update(job.id, stage="Loading model")
        self.assertGreater(job.stage_started_at, original)

    def test_running_stage_elapsed_time_is_computed_server_side(self) -> None:
        job = self.make_job()
        job.status = JobStatus.RUNNING
        job.stage_started_at = datetime.now(timezone.utc) - timedelta(seconds=75)

        self.assertGreaterEqual(job.stage_elapsed_seconds, 75)

    def test_audio_normalization_is_published_atomically(self) -> None:
        job = self.make_job()
        target = self.directory / f"{job.id}.wav"
        observed: dict[str, object] = {}

        def fake_run(command: list[str], **_kwargs) -> None:
            temporary = Path(command[-1])
            observed["temporary"] = temporary
            observed["target_existed_during_conversion"] = target.exists()
            temporary.write_bytes(b"RIFF" + b"0" * 256)

        with (
            patch("app.transcription.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("app.transcription.subprocess.run", side_effect=fake_run),
        ):
            result = _normalized_audio_path(job)

        self.assertEqual(result, target)
        self.assertFalse(observed["target_existed_during_conversion"])
        self.assertEqual(target.read_bytes(), b"RIFF" + b"0" * 256)
        self.assertFalse(Path(observed["temporary"]).exists())

    def test_diarization_runs_concurrently_with_transcription(self) -> None:
        job = self.make_job(diarization=True)
        test_case = self
        diarization_started = threading.Event()

        class FakeWhisperModel:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def transcribe(self, *_args, **_kwargs):
                def segments():
                    # Diarization is submitted before model loading even starts, so by the
                    # time transcription yields its first segment it must already be running
                    # on its own thread rather than waiting for transcription to finish.
                    test_case.assertTrue(
                        diarization_started.wait(timeout=2),
                        "diarization should start concurrently with transcription, not after it",
                    )
                    yield SimpleNamespace(start=0.0, end=3.0, text=" Raw transcript", words=[])

                return segments(), SimpleNamespace(duration=3.0, language="en")

        def fake_diarize(_audio_path: Path, current_job: Job) -> list[DiarizationTurn]:
            diarization_started.set()
            return [DiarizationTurn(0.0, 3.0, "SPEAKER_00")]

        fake_module = SimpleNamespace(WhisperModel=FakeWhisperModel)
        with (
            patch.dict(sys.modules, {"faster_whisper": fake_module}),
            patch("app.transcription._normalized_audio_path", return_value=self.upload),
            patch("app.transcription.diarize_audio", side_effect=fake_diarize),
        ):
            transcribe_job(job.id, self.store)

        self.assertTrue((self.directory / f"{job.id}.txt").read_text().find("Raw transcript") >= 0)
        self.assertTrue(job.exports_ready)
        self.assertEqual(job.status, JobStatus.DONE)
        self.assertEqual(job.segments[0].speaker, "SPEAKER_00")


if __name__ == "__main__":
    unittest.main()
