from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import re

from fastapi.testclient import TestClient

import app.main as main_module
from app.jobs import JobStatus, JobStore
from tests.test_editing import make_job


class TranscriptEditorApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.original_store = main_module.store
        main_module.store = JobStore()
        main_module.store.add(make_job(self.directory))
        self.addCleanup(self._restore_store)
        self.client = TestClient(main_module.app)

    def _restore_store(self) -> None:
        main_module.store = self.original_store

    def test_audio_endpoint_serves_the_job_upload(self) -> None:
        response = self.client.get("/jobs/job-1/audio")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"RIFF-test-audio")

    def test_job_page_preserves_submitted_transcription_options(self) -> None:
        job = main_module.store.get("job-1")
        job.options.model_size = "medium"
        job.options.language = None
        job.options.compute_type = "float32"
        job.options.vad_filter = False
        job.options.word_timestamps = True
        job.options.diarization = True
        job.options.num_speakers = 3

        response = self.client.get("/jobs/job-1")

        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertRegex(html, re.compile(r'<option\s+value="medium"\s+selected', re.MULTILINE))
        self.assertRegex(html, re.compile(r'<option value="auto"\s+selected', re.MULTILINE))
        self.assertRegex(html, re.compile(r'<option\s+value="float32"\s+selected', re.MULTILINE))
        self.assertNotRegex(html, r'name="vad_filter" checked')
        self.assertRegex(html, r'name="word_timestamps" checked')
        self.assertRegex(html, r'name="diarization" checked')
        self.assertRegex(html, re.compile(r'name="num_speakers"[\s\S]+?value="3"'))
        self.assertIn("data-stage-elapsed", html)
        self.assertRegex(html, r'/static/app\.js\?v=[a-f0-9]{12}')
        self.assertEqual(response.headers["cache-control"], "no-store")

        panel = self.client.get("/jobs/job-1/panel")
        self.assertEqual(panel.headers["cache-control"], "no-store")

    def test_bulk_edit_regenerates_exports(self) -> None:
        response = self.client.put(
            "/jobs/job-1/segments",
            json={"edits": [{"index": 0, "text": "Edited through the API"}]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Edited through the API", (self.directory / "job-1.txt").read_text())
        download = self.client.get("/jobs/job-1/download/txt")
        self.assertEqual(download.status_code, 200)
        self.assertIn("Edited through the API", download.text)

    def test_split_and_merge_return_focus_index(self) -> None:
        split = self.client.post(
            "/jobs/job-1/segments/0/split",
            json={"character_offset": 5, "text": "Hello world"},
        )
        self.assertEqual(split.status_code, 200)
        self.assertEqual(split.json()["focus_index"], 1)
        merge = self.client.post(
            "/jobs/job-1/segments/1/merge",
            json={"direction": "previous"},
        )
        self.assertEqual(merge.status_code, 200)
        self.assertEqual(merge.json()["focus_index"], 0)

    def test_editing_a_running_job_is_rejected(self) -> None:
        main_module.store.get("job-1").status = JobStatus.RUNNING
        response = self.client.put(
            "/jobs/job-1/segments",
            json={"edits": [{"index": 0, "text": "Too early"}]},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("after the job is complete", response.json()["detail"])

    def test_preliminary_exports_can_be_downloaded_while_diarization_runs(self) -> None:
        job = main_module.store.get("job-1")
        job.status = JobStatus.RUNNING
        job.exports_ready = True
        (self.directory / "job-1.txt").write_text("Preliminary transcript\n")

        response = self.client.get("/jobs/job-1/download/txt")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "Preliminary transcript\n")


if __name__ == "__main__":
    unittest.main()
