from __future__ import annotations

import re
from collections.abc import Iterable

from app.jobs import Job, Segment


class TranscriptEditError(ValueError):
    pass


def update_segment_texts(job: Job, edits: Iterable[tuple[int, str]]) -> int:
    prepared: list[tuple[int, str]] = []
    seen: set[int] = set()

    for index, text in edits:
        _segment_at(job, index)
        cleaned = _clean_text(text)
        if index in seen:
            raise TranscriptEditError(f"Segment {index + 1} was included more than once.")
        seen.add(index)
        prepared.append((index, cleaned))

    changed = 0
    for index, cleaned in prepared:
        segment = job.segments[index]
        if segment.text != cleaned:
            segment.text = cleaned
            # Word tokens no longer align after a manual text edit.
            segment.words = []
            changed += 1
    return changed


def rename_speaker(job: Job, current: str, replacement: str) -> int:
    current = current.strip()
    replacement = replacement.strip()
    if not current:
        raise TranscriptEditError("Choose a speaker to rename.")
    if not replacement:
        raise TranscriptEditError("Speaker name cannot be empty.")
    if len(replacement) > 80:
        raise TranscriptEditError("Speaker name cannot exceed 80 characters.")

    changed = 0
    for segment in job.segments:
        if segment.speaker == current:
            segment.speaker = replacement
            changed += 1
    for turn in job.diarization_turns:
        if turn.speaker == current:
            turn.speaker = replacement

    if not changed:
        raise TranscriptEditError(f'Speaker "{current}" was not found.')
    return changed


def split_segment(job: Job, index: int, character_offset: int, text: str | None = None) -> int:
    segment = _segment_at(job, index)
    source_text = segment.text if text is None else _clean_text(text)
    if character_offset <= 0 or character_offset >= len(source_text):
        raise TranscriptEditError("Place the cursor between two characters before splitting.")

    left_text = source_text[:character_offset].strip()
    right_text = source_text[character_offset:].strip()
    if not left_text or not right_text:
        raise TranscriptEditError("Both sides of a split must contain text.")

    ratio = character_offset / len(source_text)
    split_time = segment.start + max(segment.end - segment.start, 0.0) * ratio
    left_words: list[dict[str, object]] = []
    right_words: list[dict[str, object]] = []

    if text is None or source_text == segment.text:
        left_words, right_words = _partition_words(segment, split_time)
        if left_words and right_words:
            left_end = float(left_words[-1].get("end", split_time))
            right_start = float(right_words[0].get("start", split_time))
            split_time = max(segment.start, min(segment.end, (left_end + right_start) / 2))

    left = Segment(
        start=segment.start,
        end=split_time,
        text=left_text,
        speaker=segment.speaker,
        words=left_words,
    )
    right = Segment(
        start=split_time,
        end=segment.end,
        text=right_text,
        speaker=segment.speaker,
        words=right_words,
    )
    job.segments[index : index + 1] = [left, right]
    return index + 1


def merge_segment(job: Job, index: int, direction: str) -> int:
    _segment_at(job, index)
    if direction == "previous":
        if index == 0:
            raise TranscriptEditError("The first segment has no previous segment.")
        left_index, right_index = index - 1, index
    elif direction == "next":
        if index >= len(job.segments) - 1:
            raise TranscriptEditError("The last segment has no next segment.")
        left_index, right_index = index, index + 1
    else:
        raise TranscriptEditError("Merge direction must be previous or next.")

    left = job.segments[left_index]
    right = job.segments[right_index]
    if left.speaker != right.speaker:
        raise TranscriptEditError("Rename the speakers to match before merging these segments.")

    merged = Segment(
        start=min(left.start, right.start),
        end=max(left.end, right.end),
        text=f"{left.text.rstrip()} {right.text.lstrip()}".strip(),
        speaker=left.speaker,
        words=[*left.words, *right.words],
    )
    job.segments[left_index : right_index + 1] = [merged]
    return left_index


def replace_text(job: Job, search: str, replacement: str, match_case: bool = False) -> int:
    if not search:
        raise TranscriptEditError("Search text cannot be empty.")
    if len(search) > 1_000 or len(replacement) > 10_000:
        raise TranscriptEditError("Search or replacement text is too long.")

    flags = 0 if match_case else re.IGNORECASE
    pattern = re.compile(re.escape(search), flags)
    prepared: list[tuple[Segment, str, int]] = []
    total = 0

    for segment in job.segments:
        updated, count = pattern.subn(lambda _match: replacement, segment.text)
        if count and not updated.strip():
            raise TranscriptEditError("Replace all would leave a segment empty.")
        prepared.append((segment, updated, count))
        total += count

    for segment, updated, count in prepared:
        if count:
            segment.text = updated
            segment.words = []
    return total


def _segment_at(job: Job, index: int) -> Segment:
    if index < 0 or index >= len(job.segments):
        raise TranscriptEditError("Segment was not found. Refresh the page and try again.")
    return job.segments[index]


def _clean_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise TranscriptEditError("Segment text cannot be empty.")
    if len(cleaned) > 100_000:
        raise TranscriptEditError("Segment text cannot exceed 100,000 characters.")
    return cleaned


def _partition_words(
    segment: Segment, split_time: float
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(segment.words) < 2:
        return [], []
    left = [word for word in segment.words if float(word.get("end", segment.start)) <= split_time]
    right = [word for word in segment.words if float(word.get("end", segment.start)) > split_time]
    if not left or not right:
        return [], []
    return left, right
