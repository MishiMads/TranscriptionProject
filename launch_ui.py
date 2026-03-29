import os, sys, glob

# Add pip-installed NVIDIA CUDA DLL directories to the search path.
# These are installed by nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, etc.
_nvidia_bin_dirs = glob.glob(
    os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")
)
for _dir in _nvidia_bin_dirs:
    os.add_dll_directory(_dir)

from transcriber.ui import build_ui
import gradio as gr

demo = build_ui()
demo.launch(theme=gr.themes.Soft())
