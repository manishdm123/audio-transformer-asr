from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.editing import (
    TranscriptEditError,
    merge_segment,
    rename_speaker,
    replace_text,
    split_segment,
    update_segment_texts,
)
from app.jobs import DiarizationTurn, Job, JobStatus, Segment, TranscriptionOptions
from app.transcription import _srt_time, write_exports


def make_job(directory: Path) -> Job:
    upload = directory / "meeting.wav"
    upload.write_bytes(b"RIFF-test-audio")
    return Job(
        id="job-1",
        filename="meeting.wav",
        upload_path=upload,
        output_dir=directory,
        options=TranscriptionOptions("small", "en", "int8", True, True, diarization=True),
        status=JobStatus.DONE,
        stage="Complete",
        progress=1.0,
        segments=[
            Segment(0.0, 4.0, "Hello world", "SPEAKER_00"),
            Segment(4.0, 8.0, "hello again", "SPEAKER_00"),
            Segment(8.0, 12.0, "A response", "SPEAKER_01"),
        ],
        diarization_turns=[
            DiarizationTurn(0.0, 8.0, "SPEAKER_00"),
            DiarizationTurn(8.0, 12.0, "SPEAKER_01"),
        ],
    )


class TranscriptEditingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.job = make_job(self.directory)

    def test_bulk_text_update_is_validated_before_changes_are_applied(self) -> None:
        with self.assertRaises(TranscriptEditError):
            update_segment_texts(self.job, [(0, "Changed"), (99, "Missing")])
        self.assertEqual(self.job.segments[0].text, "Hello world")

    def test_split_uses_cursor_position_and_preserves_timeline(self) -> None:
        new_index = split_segment(self.job, 0, 5, "Hello world")
        self.assertEqual(new_index, 1)
        self.assertEqual([segment.text for segment in self.job.segments[:2]], ["Hello", "world"])
        self.assertEqual(self.job.segments[0].end, self.job.segments[1].start)
        self.assertAlmostEqual(self.job.segments[0].end, 4.0 * (5 / 11))

    def test_merge_requires_matching_speakers(self) -> None:
        merged_index = merge_segment(self.job, 1, "previous")
        self.assertEqual(merged_index, 0)
        self.assertEqual(self.job.segments[0].text, "Hello world hello again")
        with self.assertRaises(TranscriptEditError):
            merge_segment(self.job, 0, "next")

    def test_rename_speaker_updates_segments_and_diarization_turns(self) -> None:
        changed = rename_speaker(self.job, "SPEAKER_00", "Manish")
        self.assertEqual(changed, 2)
        self.assertEqual(self.job.segments[0].speaker, "Manish")
        self.assertEqual(self.job.diarization_turns[0].speaker, "Manish")

    def test_replace_all_is_case_insensitive_by_default(self) -> None:
        count = replace_text(self.job, "hello", "Hi")
        self.assertEqual(count, 2)
        self.assertEqual(self.job.segments[0].text, "Hi world")
        self.assertEqual(self.job.segments[1].text, "Hi again")

    def test_export_regeneration_reflects_edits(self) -> None:
        rename_speaker(self.job, "SPEAKER_00", "Manish")
        update_segment_texts(self.job, [(0, "Updated introduction")])
        write_exports(self.job)
        self.assertIn("Manish:", (self.directory / "job-1.txt").read_text())
        self.assertIn("Updated introduction", (self.directory / "job-1.md").read_text())
        self.assertIn('"text": "Updated introduction"', (self.directory / "job-1.json").read_text())

    def test_srt_timestamp_rounding_carries_to_the_next_second(self) -> None:
        self.assertEqual(_srt_time(59.9996), "00:01:00,000")


if __name__ == "__main__":
    unittest.main()
