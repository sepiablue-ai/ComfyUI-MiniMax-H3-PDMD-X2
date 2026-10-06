"""Install into the selected ComfyUI Python and prepare the CPU-fused checkpoint."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from common import ROOT, free_bytes, load_config, recipe, require_comfy_python, verify_model, verify_sources


def install_dependencies(config):
    core = config["comfyui_root"]
    # The CUDA build is chosen by the user when installing ComfyUI. Never let
    # the general dependency resolver replace it with a different build.
    protected = []
    for package in ("torch", "torchvision", "torchaudio"):
        try:
            protected.append(f"{package}=={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            if package in ("torch", "torchvision"):
                raise RuntimeError("Install CUDA PyTorch and torchvision with this ComfyUI Python first; see README")
    x2 = core / "custom_nodes" / "ComfyUI-H3-X2-Stream"
    if not x2.exists():
        if not shutil.which("git"):
            raise RuntimeError("Git is required to install ComfyUI-H3-X2-Stream")
        subprocess.run(["git", "clone", "--no-checkout", recipe()["sources"]["x2_stream"], str(x2)], check=True)
        subprocess.run(["git", "-C", str(x2), "checkout", recipe()["x2_stream_commit"]], check=True)
    if not (x2 / "__init__.py").is_file() or not (x2 / "requirements.txt").is_file():
        raise RuntimeError(f"Incomplete existing X2 installation; inspect it before retrying: {x2}")
    requirements = [core / "requirements.txt", x2 / "requirements.txt"]
    if not all(path.is_file() for path in requirements):
        raise FileNotFoundError("ComfyUI requirements.txt was not found")
    scratch = core / "tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdmd-setup-", dir=scratch) as temporary:
        constraints = Path(temporary) / "constraints.txt"
        constraints.write_text("\n".join(protected) + "\n", encoding="utf-8")
        uv = shutil.which("uv")
        command = ([uv, "pip", "install", "--python", sys.executable] if uv else
                   [sys.executable, "-m", "pip", "install"])
        command += ["--constraint", str(constraints)]
        for path in requirements:
            command += ["--requirement", str(path)]
        subprocess.run(command, check=True)
    return x2


def copy_sample_images(config):
    destination = config["comfyui_root"] / "input"
    destination.mkdir(exist_ok=True)
    for source in sorted((ROOT / "examples" / "assets").glob("*.png")):
        target = destination / source.name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
                raise RuntimeError(f"An unrelated input image already exists; keep it and resolve the name conflict: {target}")
        else:
            shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true", help="Read-only preflight; no fusion, copies or writes")
    args = parser.parse_args()
    config = load_config(args.config)
    require_comfy_python(config)
    # Check source sizes and core compatibility before changing the selected venv.
    verify_sources(config, check_hash=False, check_kernel=False)
    if not args.check:
        install_dependencies(config)
    for name in ("torch", "safetensors", "psutil", "comfy_kitchen", "av"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f"Missing {name} in the selected ComfyUI environment; run setup without --check to install dependencies")
    x2 = config["comfyui_root"] / "custom_nodes" / "ComfyUI-H3-X2-Stream"
    if not (x2 / "__init__.py").is_file():
        raise FileNotFoundError("Install ComfyUI-H3-X2-Stream into this ComfyUI; see README")
    verify_sources(config)
    model, manifest = config["model_path"], config["fusion_manifest"]
    reused = model.is_file()
    if reused:
        verify_model(model, manifest)
    elif model.exists() or manifest.exists() or model.with_suffix(".safetensors.partial").exists():
        raise RuntimeError("Incomplete existing preparation; inspect it before retrying")
    elif free_bytes(config["output_directory"]) < 30 * 10 ** 9:
        raise RuntimeError("Allow at least 30 GB free on the destination volume for fusion")
    if not args.check and not reused:
        subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "scripts" / "fuse_pdmd_cpu.py"),
                        "--config", str(config["config_path"])], check=True)
        verify_model(model, manifest)
    if not args.check:
        copy_sample_images(config)
        # Generate a separate config; never modify an existing ComfyUI configuration.
        path = config["output_directory"].as_posix()
        yaml = "pdmd_x2_fused:\n  diffusion_models: " + json.dumps(path, ensure_ascii=False) + "\n"
        (config["config_path"].parent / "extra_model_paths.local.yaml").write_text(yaml, encoding="utf-8")
    print(json.dumps({"status": "check_passed" if args.check else "model_ready",
                      "existing_model_reused": reused, "model": str(model),
                      "generation_started": False, "comfyui_started": False,
                      "dependencies_installed": not args.check,
                      "sample_images_prepared": not args.check, "read_only": args.check}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
