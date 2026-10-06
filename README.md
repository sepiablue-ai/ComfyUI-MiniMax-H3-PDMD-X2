# ComfyUI MiniMax H3 PDMD X2

[日本語](README_ja.md) | English

Generate MiniMax H3 first-frame video with a fused **PDMD two-step** model, **SLA keep 20%**, and **X2 VAE INT8 streaming**. The examples sample at **544 × 960** and save **1088 × 1920**, **124 frames at 24 fps**, with generated audio.

This package supplies installation, CPU model preparation and ready-to-load workflows. Keep it outside `ComfyUI/custom_nodes`. Setup installs the separate [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) dependency into your selected ComfyUI.

The corresponding comparison is **FL2VA 015** on [Sayaka Benchmark](https://sayakabenchmark.pages.dev/). **Known limitation:** generated audio can be poor with this two-step preset.

## Included files

- `setup.bat` and `scripts/`: install dependencies, prepare sample images, and fuse the PDMD LoRA into the base model on CPU.
- `config.example.json`: model paths to fill in for your installation.
- `model_recipe.json`: accepted input model hashes and native fusion implementation hashes.
- `workflows/`: two ComfyUI GUI examples and their API graphs.
- `examples/assets/`: the images used by the examples.
- `LICENSE` and `NOTICE.md`: code license, attribution and separate model terms.

## Requirements

- Windows and an existing ComfyUI with native MiniMax H3 and INT8 ConvRot support. Tested core: ComfyUI 0.38.0, commit `6b747c0428c343e1417219641db93a4fb7cb69ae`. Preparation checks the relevant native source hashes and stops on an incompatible implementation.
- The Python from that ComfyUI's venv or Windows Portable installation. Tested on a fresh Windows clone and venv with Python 3.13.13, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, comfy-kitchen 0.2.36, safetensors 0.8.0, psutil 7.2.2, PyAV 19.0.1 and transformers 5.18.0.
- Git on PATH for installing ComfyUI-H3-X2-Stream. Setup installs revision `eb148a3d0f91a273668bbe027bfcbafb85f2c5e8` when the node folder is absent; it keeps an existing installation. Package installation uses `uv` when available, otherwise the selected Python's `pip`.
- For generation: an NVIDIA CUDA GPU and working H.264 NVENC support. Tested on RTX 4070 12 GB; other devices and operating systems have not been validated.
- At least **30 GB free** on the fusion destination volume, in addition to the existing source models. The fused model occupies about **21 GB**. Fusion uses CPU only, two threads, and waits when less than 8 GiB system RAM is available.

Setup installs ComfyUI's requirements and the X2 node's requirements into the selected ComfyUI environment. Install CUDA PyTorch and torchvision there first; setup keeps the installed torch/torchvision/torchaudio versions. Model weights must be provided separately. Setup does not change ComfyUI core or start a server.

For a new Windows installation, use the tested core revision and create its venv first. For example, with Git and uv installed:

```bat
git clone https://github.com/Comfy-Org/ComfyUI.git C:\ComfyUI
git -C C:\ComfyUI checkout 6b747c0428c343e1417219641db93a4fb7cb69ae
uv venv --seed --python 3.13 C:\ComfyUI\.venv
uv pip install --python C:\ComfyUI\.venv\Scripts\python.exe torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

These CUDA packages are the tested Windows builds. Choose a compatible CUDA build according to the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) if your system requires another build.

## Models

Download the required models according to their respective terms. No weights are bundled.

| File | ComfyUI folder | Role |
|---|---|---|
| [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) | `models/diffusion_models/` | Exact unfused base for preparation |
| [minimax_h3_pdmd_2nfe_comfyui.safetensors](https://huggingface.co/Iwannapose/minimax_h3_pdmd_2nfe_comfyui) | `models/loras/` | Converted BF16 PDMD 2-NFE LoRA, strength 1.0 |
| [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) | `models/text_encoders/` | Text encoder |
| [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `models/vae/` | First-frame conditioning |
| [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `models/vae/` | Audio decode |
| [MiniMax-H3-X2-Detail-v1.safetensors](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE) | `models/vae/` | Source for X2 INT8 preparation |

The bundled samples contain **H3 X2 Prepare INT8 VAE**. Select the original X2 Detail v1 model there. The first queue converts it into `models/vae/h3_x2_stream/MiniMax-H3-X2-Detail-v1-decoder-int8-convrot.safetensors`; later queues reuse the converted file. See the dependency's [model-preparation instructions](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream#generate-in-comfyui).

## Prepare the PDMD model

1. Copy `config.example.json` to `config.local.json`.
2. Set `comfyui_root` to the folder containing `main.py` and `comfy/`. Set `base_model` and `pdmd_lora` to your existing files. Choose `output_directory` for the fused model; relative paths resolve from the config file's folder. Leave `fusion_manifest` null to store it beside the output model.
3. Run with your **ComfyUI Python**. For a venv installation:

```bat
setup.bat "C:\ComfyUI\.venv\Scripts\python.exe" "config.local.json"
setup.bat "C:\ComfyUI\.venv\Scripts\python.exe" "config.local.json" --check
```

For Windows Portable, replace the Python path with that installation's `python_embeded\python.exe`.

Normal setup installs the dependencies, then reuses a completed output only when its manifest and hash match; otherwise it creates `minimax_h3_pdmd_2nfe_fused_int8_convrot.safetensors` on CPU. It fuses 200 INT8 ConvRot modules and eight BF16 modules, verifies changed tensors and unchanged data, and preserves the base and LoRA. Existing unrelated or incomplete outputs cause an error. `--check` is a read-only preflight for an already installed environment; it does not install missing packages or nodes.

4. Setup writes `extra_model_paths.local.yaml` beside the config file. Add that section to your ComfyUI `extra_model_paths.yaml`, or pass the file through ComfyUI's `--extra-model-paths-config` option. If `output_directory` is already ComfyUI's `models/diffusion_models`, it is found automatically. Start ComfyUI through your normal launcher.

The preparation manifest and progress file are written alongside the configured model output, not into the distributed package. Preparation and video generation are separate operations.

## Generate in ComfyUI

1. Setup copies the PNGs from [examples/assets](examples/assets) into `ComfyUI/input/`. It keeps identical existing images and refuses to overwrite different files with the same names.
2. Drag [case01](workflows/fl2va_015_case01.json) or [case02](workflows/fl2va_015_case02.json) onto ComfyUI. API graphs are in [workflows/api](workflows/api).
3. Select the fused PDMD model, text encoder, ordinary H3 video VAE, audio VAE and original X2 Detail v1 model.
4. Set the stream-save node's NVENC GPU index for your installation, then queue.

**Do not apply the PDMD LoRA again to the fused model.** Keep the ordinary H3 video VAE on the image-conditioning branch and the X2 decoder on the final video branch.

Preset: Euler/simple, two steps, BasicGuider, seed 43, video/audio shifts 12/3, SLA keep 20%, FFN chunks 8, NVENC CQ 18 and `fp16_accumulation=true`. Disable the last option if your PyTorch build lacks the required support. Videos use the configured ComfyUI output directory under `FL2VA_015_PDMD_X2/`.

## Troubleshooting

- Source hash or native implementation mismatch: use the exact models and tested fusion implementation listed in `model_recipe.json`. Another PDMD or base checkpoint is not interchangeable.
- Missing X2 nodes: run setup with your ComfyUI Python and restart ComfyUI. The sample's preparation node creates the X2 INT8 decoder on the first queue.
- Fused model missing from the selector: verify the model-path configuration and restart ComfyUI.
- Incomplete fusion output: inspect the configured destination before retrying. Setup refuses to overwrite a partial checkpoint or existing manifest.
- Poor audio: this is a known limitation of the supplied two-step preset. CPU fusion and requantization can also change video/audio compared with applying the LoRA dynamically.

For full CPU verification of an existing output, run your ComfyUI Python with `-B scripts/fuse_pdmd_cpu.py --config config.local.json --verify`. This reads the base, LoRA and output without rebuilding.

## License

GPL-3.0 code and workflow adaptations. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Model weights remain subject to their respective model terms.
