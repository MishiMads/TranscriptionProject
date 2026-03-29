"""Gradio web UI for the multilingual transcriber."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from .exporters import write_json, write_srt, write_txt
from .pipeline import TranscriberConfig, TranscriberService

_MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]
_LANGUAGE_CHOICES = {
    "Auto-detect": "auto",
    "Danish (da)": "da",
    "English (en)": "en",
    "German (de)": "de",
    "Spanish (es)": "es",
    "French (fr)": "fr",
    "Italian (it)": "it",
    "Japanese (ja)": "ja",
    "Korean (ko)": "ko",
    "Chinese (zh)": "zh",
    "Portuguese (pt)": "pt",
    "Russian (ru)": "ru",
    "Dutch (nl)": "nl",
    "Polish (pl)": "pl",
    "Swedish (sv)": "sv",
    "Norwegian (nb)": "nb",
    "Finnish (fi)": "fi",
    "Arabic (ar)": "ar",
    "Hindi (hi)": "hi",
    "Turkish (tr)": "tr",
}

# Cache services by (model_size, compute_type) to avoid reloading the model
_service_cache: dict[tuple[str, str], TranscriberService] = {}

# Single app-level temp directory so output files persist until the process exits
_tmp_dir = Path(tempfile.mkdtemp(prefix="transcriber_ui_"))


def _get_service(model_size: str, compute_type: str) -> TranscriberService:
    key = (model_size, compute_type)
    if key not in _service_cache:
        config = TranscriberConfig(model_size=model_size, device="auto", compute_type=compute_type)
        _service_cache[key] = TranscriberService(config)
    return _service_cache[key]


def _transcribe(
    audio_file: str | None,
    model_size: str,
    language_label: str,
    task: str,
    output_format: list[str],
    beam_size: int,
    vad_filter: bool,
    compute_type: str,
) -> tuple[str, str, str | None, str | None, str | None]:
    """Run transcription and return (transcript, info, txt_path, srt_path, json_path)."""

    if audio_file is None:
        return "", "⚠️ Please upload an audio file.", None, None, None

    language_code = _LANGUAGE_CHOICES.get(language_label, "auto")
    language_option = None if language_code == "auto" else language_code

    service = _get_service(model_size, compute_type)

    audio_path = Path(audio_file)
    result = service.transcribe(
        audio_path=audio_path,
        language=language_option,
        task=task,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    formats = {fmt.lower() for fmt in output_format}

    base = _tmp_dir / audio_path.stem

    txt_path: str | None = None
    srt_path: str | None = None
    json_path: str | None = None

    if "txt" in formats:
        p = base.with_suffix(".txt")
        write_txt(result, p)
        txt_path = str(p)

    if "srt" in formats:
        p = base.with_suffix(".srt")
        write_srt(result, p)
        srt_path = str(p)

    if "json" in formats:
        p = base.with_suffix(".json")
        write_json(result, p)
        json_path = str(p)

    duration_str = (
        f"{result.duration_seconds:.1f}s" if result.duration_seconds else "unknown"
    )
    info = (
        f"✅ Done — Detected language: **{result.language}** "
        f"({result.language_probability:.1%} confidence) | "
        f"Duration: {duration_str} | "
        f"Segments: {len(result.segments)}"
    )

    return result.full_text, info, txt_path, srt_path, json_path


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Multilingual Transcriber") as demo:
        gr.Markdown(
            """
# 🎙️ Multilingual Transcriber
Upload an audio or video file and click **Transcribe** to generate a transcript.
Powered by [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper).
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload audio / video file",
                    type="filepath",
                    sources=["upload"],
                )

                with gr.Accordion("⚙️ Settings", open=False):
                    model_size = gr.Dropdown(
                        choices=_MODEL_CHOICES,
                        value="medium",
                        label="Model size",
                        info="Larger models are more accurate but slower.",
                    )
                    language = gr.Dropdown(
                        choices=list(_LANGUAGE_CHOICES.keys()),
                        value="Auto-detect",
                        label="Language",
                    )
                    task = gr.Radio(
                        choices=["transcribe", "translate"],
                        value="transcribe",
                        label="Task",
                        info="'translate' converts speech to English text.",
                    )
                    output_format = gr.CheckboxGroup(
                        choices=["txt", "srt", "json"],
                        value=["txt", "srt", "json"],
                        label="Output formats",
                    )
                    beam_size = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=5,
                        label="Beam size",
                        info="Higher = slightly more accurate, but slower.",
                    )
                    vad_filter = gr.Checkbox(
                        value=True,
                        label="VAD filter (remove silence)",
                    )
                    compute_type = gr.Dropdown(
                        choices=["int8", "float16", "float32"],
                        value="int8",
                        label="Compute type",
                        info="int8 is fastest; float32 is most precise.",
                    )

                transcribe_btn = gr.Button("▶ Transcribe", variant="primary")

            with gr.Column(scale=2):
                status_md = gr.Markdown(value="")
                transcript_box = gr.Textbox(
                    label="Transcript",
                    lines=20,
                    max_lines=40,
                    interactive=False,
                )
                with gr.Row():
                    txt_download = gr.File(label="Download .txt", visible=False)
                    srt_download = gr.File(label="Download .srt", visible=False)
                    json_download = gr.File(label="Download .json", visible=False)

        def run_transcription(
            audio_file,
            model_size,
            language,
            task,
            output_format,
            beam_size,
            vad_filter,
            compute_type,
        ):
            transcript, info, txt, srt, json_ = _transcribe(
                audio_file,
                model_size,
                language,
                task,
                output_format,
                beam_size,
                vad_filter,
                compute_type,
            )
            return (
                transcript,
                info,
                gr.update(value=txt, visible=txt is not None),
                gr.update(value=srt, visible=srt is not None),
                gr.update(value=json_, visible=json_ is not None),
            )

        transcribe_btn.click(
            fn=run_transcription,
            inputs=[
                audio_input,
                model_size,
                language,
                task,
                output_format,
                beam_size,
                vad_filter,
                compute_type,
            ],
            outputs=[transcript_box, status_md, txt_download, srt_download, json_download],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
