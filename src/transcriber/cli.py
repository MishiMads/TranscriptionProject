from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from .exporters import write_json, write_srt, write_txt
from .pipeline import TranscriberConfig, TranscriberService

app = typer.Typer(help="Transcribe long multilingual recordings (Danish, English, and more).")


@app.command()
def transcribe(
    input_file: Annotated[Path, typer.Argument(exists=True, readable=True, help="Path to recording")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Output folder")] = Path("outputs"),
    language: Annotated[
        str,
        typer.Option(
            "--language",
            "-l",
            help="Language code (for example 'da' or 'en'). Use 'auto' for detection.",
        ),
    ] = "auto",
    model_size: Annotated[
        str,
        typer.Option("--model", "-m", help="Model size: tiny, base, small, medium, large-v3"),
    ] = "medium",
    device: Annotated[str, typer.Option(help="auto, cpu, or cuda")] = "auto",
    compute_type: Annotated[str, typer.Option(help="int8, float16, float32")] = "int8",
    task: Annotated[Literal["transcribe", "translate"], typer.Option(help="transcribe or translate")] = "transcribe",
    beam_size: Annotated[int, typer.Option(help="Beam size for decoding")] = 5,
    vad_filter: Annotated[bool, typer.Option(help="Filter silence before decoding")] = True,
    output_format: Annotated[
        list[str] | None,
        typer.Option(
            "--format",
            "-f",
            help="Output formats. Repeat flag for multiple: --format txt --format srt --format json",
        ),
    ] = None,
) -> None:
    language_option = None if language.lower() == "auto" else language.lower()

    config = TranscriberConfig(model_size=model_size, device=device, compute_type=compute_type)
    service = TranscriberService(config)

    result = service.transcribe(
        audio_path=input_file,
        language=language_option,
        task=task,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / input_file.stem

    formats = {fmt.lower().strip() for fmt in (output_format or ["txt", "srt", "json"])}
    if "txt" in formats:
        write_txt(result, base_path.with_suffix(".txt"))
    if "srt" in formats:
        write_srt(result, base_path.with_suffix(".srt"))
    if "json" in formats:
        write_json(result, base_path.with_suffix(".json"))

    typer.echo(f"Detected language: {result.language} ({result.language_probability:.2%})")
    typer.echo(f"Segments written: {len(result.segments)}")
    typer.echo(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    app()

