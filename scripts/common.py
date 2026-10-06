"""Configuration and bounded I/O; importing this module does not load torch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "minimax_h3_pdmd_2nfe_fused_int8_convrot.safetensors"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_config(path):
    path = Path(path).resolve(strict=True)
    data = read_json(path)
    for key in ("comfyui_root", "base_model", "pdmd_lora", "output_directory"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"Set {key} in {path.name}")
    for key in ("comfyui_root", "base_model", "pdmd_lora", "output_directory",
                "fusion_manifest"):
        if data.get(key):
            candidate = Path(data[key]).expanduser()
            data[key] = (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
        else:
            data[key] = None
    data["config_path"] = path
    data["model_path"] = data["output_directory"] / MODEL_NAME
    data["fusion_manifest"] = data["fusion_manifest"] or data["output_directory"] / "fusion_manifest.json"
    return data


def require_comfy_python(config):
    core = config["comfyui_root"]
    executable = Path(sys.executable).resolve()
    candidates = [folder / ".venv" / "Scripts" / "python.exe"
                  for folder in (core, core.parent)]
    candidates += [core.parent / "python_embeded" / "python.exe"]
    candidates += [folder / ".venv" / "bin" / "python" for folder in (core, core.parent)]
    if executable not in [p.resolve() for p in candidates if p.is_file()]:
        raise RuntimeError("Use the Python from the configured ComfyUI venv or Portable installation")
    if not (core / "comfy" / "lora.py").is_file():
        raise FileNotFoundError(f"ComfyUI core was not found: {core}")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_header(path):
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Incomplete safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= 100 * 1024 ** 2:
            raise ValueError("Invalid safetensors header size")
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError("Incomplete safetensors header")
    return json.loads(raw), 8 + size, prefix + raw


def free_bytes(directory):
    directory = Path(directory)
    while not directory.exists():
        parent = directory.parent
        if parent == directory:
            raise FileNotFoundError(f"Destination volume/share is unavailable: {directory}")
        directory = parent
    return shutil.disk_usage(directory).free


def recipe():
    return read_json(ROOT / "model_recipe.json")


def verify_sources(config, check_hash=True, check_kernel=True):
    reference = recipe()
    for field, name in (("base_model", "base"), ("pdmd_lora", "lora")):
        path = config[field]
        expected = reference[name]
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing or wrong-size {name}: {path}")
        if check_hash and sha256(path) != expected["sha256"]:
            raise ValueError(f"Wrong {name} SHA-256; use the exact unfused base and converted 2-NFE LoRA")
    for relative, expected in reference["comfyui_source_sha256"].items():
        path = config["comfyui_root"] / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Unsupported ComfyUI source revision: {relative}; see README tested revision")
    if not check_kernel:
        return
    spec = importlib.util.find_spec("comfy_kitchen")
    kernel = Path(spec.origin).parent / "tensor" / "int8.py" if spec and spec.origin else None
    if kernel is None or not kernel.is_file() or sha256(kernel) != reference["int8_source_sha256"]:
        raise ValueError("Unsupported Comfy Kitchen INT8 implementation; see README tested revision")


def verify_model(model, manifest):
    model, manifest = Path(model), Path(manifest)
    data = read_json(manifest)
    reference = recipe()
    if data.get("status") != "created_and_cpu_verified":
        raise ValueError("Fusion did not finish successfully")
    for key in ("base", "lora"):
        if data[key]["sha256"] != reference[key]["sha256"]:
            raise ValueError(f"Wrong fusion provenance: {key}")
    if float(data["lora"]["strength"]) != 1.0:
        raise ValueError("Only strength 1.0 is supported")
    if model.stat().st_size != data["output"]["bytes"]:
        raise ValueError("Fused model size does not match its manifest")
    header, _, _ = load_header(model)
    metadata = header.get("__metadata__", {})
    if metadata.get("pdmd_lora_sha256") != reference["lora"]["sha256"]:
        raise ValueError("Fused model metadata does not match the PDMD LoRA")
    if metadata.get("pdmd_lora_strength") != "1.0":
        raise ValueError("Wrong PDMD strength in fused model metadata")
    if sha256(model) != data["output"]["sha256"]:
        raise ValueError("Fused model SHA-256 does not match its manifest")
    return data
