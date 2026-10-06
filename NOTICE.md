# Attribution and model terms

Code and example workflow adaptations are distributed under GPL-3.0; see LICENSE.

- Example UI workflows are adapted from [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream), GPL-3.0. Changes: fused PDMD model, two-step Euler sampling, SLA keep 20%, preprocessed inputs, prepared X2 decoder, output labels and quality note. Its implementation is a dependency and is not bundled here.
- CPU fusion calls the native LoRA and INT8 ConvRot operations supplied by [ComfyUI](https://github.com/Comfy-Org/ComfyUI) and [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen). It does not replace their source files.
- The converted PDMD LoRA is from [Iwannapose/minimax_h3_pdmd_2nfe_comfyui](https://huggingface.co/Iwannapose/minimax_h3_pdmd_2nfe_comfyui). Original PDMD work: [pdmd2026/pdmd_2NFE_lora](https://huggingface.co/pdmd2026/pdmd_2NFE_lora).
- The base model and H3 components are from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3); applicable MiniMax H3 model terms continue to apply.
- The X2 VAE is from [speach1sdef178](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE). Its LICENSE and NOTICE continue to apply to the original and locally converted weights.
- Model weights are not included. The code license does not replace model terms.
- Example prompts and preprocessed images are supplied for the included workflows.
