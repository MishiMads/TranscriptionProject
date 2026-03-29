from __future__ import annotations

import json
from pathlib import Path

from .types import TranscriptResult


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_txt(result: TranscriptResult, output_path: Path) -> None:
    output_path.write_text(result.full_text + "\n", encoding="utf-8")


def write_srt(result: TranscriptResult, output_path: Path) -> None:
    lines: list[str] = []
    for idx, segment in enumerate(result.segments, start=1):
        lines.extend(
            [
                str(idx),
                f"{_format_srt_timestamp(segment.start)} --> {_format_srt_timestamp(segment.end)}",
                segment.text,
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(result: TranscriptResult, output_path: Path) -> None:
    payload = {
        "source_file": result.source_file,
        "language": result.language,
        "language_probability": result.language_probability,
        "duration_seconds": result.duration_seconds,
        "segments": [segment.to_dict() for segment in result.segments],
        "full_text": result.full_text,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

