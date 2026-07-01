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
- Optional speaker diarization uses `pyannote.audio`.
- Audio is stored under `data/uploads`.
- Transcript exports are generated under `data/outputs`.
- If `ffmpeg` is available, uploads are normalized to 16 kHz mono WAV before transcription.

## Speaker diarization

Diarization labels who spoke when, using generic labels such as `SPEAKER_00`.

To enable it:

1. Accept access to `pyannote/speaker-diarization-community-1` on Hugging Face.
2. Create a Hugging Face token with model read access.
3. Set the token before starting the app:

```bash
export HF_TOKEN=your_token_here
uv run audio2text
```

Or copy `.env.example` to `.env` and fill in `HF_TOKEN`.

When diarization is selected, the app automatically enables word timestamps so speakers can be merged into the transcript more accurately.
