from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .types import TranscriptResult, TranscriptSegment


@dataclass(slots=True)
class TranscriberConfig:
    model_size: str = "medium"
    device: str = "auto"
    compute_type: str = "int8"


class TranscriberService:
    def __init__(self, config: TranscriberConfig) -> None:
        self.config = config
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Run: pip install -e ."
                ) from exc

            self._model = WhisperModel(
                model_size_or_path=self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        condition_on_previous_text: bool = True,
    ) -> TranscriptResult:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._get_model()

        # faster-whisper supports long recordings by decoding in windows internally.
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=condition_on_previous_text,
        )

        parsed_segments: list[TranscriptSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            parsed_segments.append(
                TranscriptSegment(start=float(segment.start), end=float(segment.end), text=text)
            )

        return TranscriptResult(
            source_file=str(audio_path),
            language=info.language,
            language_probability=float(info.language_probability),
            duration_seconds=float(getattr(info, "duration", 0.0) or 0.0),
            segments=parsed_segments,
        )

