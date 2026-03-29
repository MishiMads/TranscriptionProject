from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class TranscriptResult:
    source_file: str
    language: str
    language_probability: float
    duration_seconds: float | None
    segments: list[TranscriptSegment]

    @property
    def full_text(self) -> str:
        return "\n".join(segment.text for segment in self.segments if segment.text)

