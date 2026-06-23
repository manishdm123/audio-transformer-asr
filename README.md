# audio2text

Local-first transcription workspace for long business conversations.

## Run

```bash
uv sync
uv run audio2text
```

Then open http://127.0.0.1:8000.

## Notes

- Transcription uses `faster-whisper` locally.
- Audio is stored under `data/uploads`.
- Transcript exports are generated under `data/outputs`.
- If `ffmpeg` is available, uploads are normalized to 16 kHz mono WAV before transcription.
- Speaker diarization is intentionally not in V1; the app keeps the backend shape ready for it.
