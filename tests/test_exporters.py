from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcriber.exporters import write_json, write_srt, write_txt
from transcriber.types import TranscriptResult, TranscriptSegment


class ExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = TranscriptResult(
            source_file="sample.wav",
            language="da",
            language_probability=0.98,
            duration_seconds=4.2,
            segments=[
                TranscriptSegment(start=0.0, end=1.2, text="Hej verden"),
                TranscriptSegment(start=1.3, end=3.0, text="Hello world"),
            ],
        )

    def test_txt_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "out.txt"
            write_txt(self.result, out)
            self.assertIn("Hej verden", out.read_text(encoding="utf-8"))

    def test_srt_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "out.srt"
            write_srt(self.result, out)
            content = out.read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:01,200", content)
            self.assertIn("Hello world", content)

    def test_json_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "out.json"
            write_json(self.result, out)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["language"], "da")
            self.assertEqual(len(payload["segments"]), 2)


if __name__ == "__main__":
    unittest.main()

