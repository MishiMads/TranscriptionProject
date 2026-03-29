from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcriber.exporters import write_json, write_srt, write_txt
from transcriber.pipeline import TranscriberConfig, TranscriberService


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick transcription runner")
    parser.add_argument("input_file", type=Path, help="Audio file path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--language", default="auto", help="Language code, ex: da or en")
    args = parser.parse_args()

    service = TranscriberService(TranscriberConfig())
    language = None if args.language == "auto" else args.language
    result = service.transcribe(audio_path=args.input_file, language=language)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_base = args.output_dir / args.input_file.stem
    write_txt(result, out_base.with_suffix(".txt"))
    write_srt(result, out_base.with_suffix(".srt"))
    write_json(result, out_base.with_suffix(".json"))
    print(f"Done. Files written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

