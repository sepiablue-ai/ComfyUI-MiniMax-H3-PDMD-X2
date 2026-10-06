"""Create a separate PDMD INT8 ConvRot checkpoint on CPU, one layer at a time."""
import os

# Set these before importing PyTorch or ComfyUI. Never expose either GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"

import argparse
import dataclasses
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import psutil
import torch
from safetensors import safe_open

from common import load_config, require_comfy_python, verify_sources, free_bytes, ROOT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', required=True)
parser.add_argument('--verify', action='store_true')
options = parser.parse_args()
CONFIG = load_config(options.config)
require_comfy_python(CONFIG)
verify_sources(CONFIG)
CORE = CONFIG['comfyui_root']
BASE = CONFIG['base_model']
LORA = CONFIG['pdmd_lora']
OUTPUT = CONFIG['model_path']
PARTIAL = OUTPUT.with_suffix('.safetensors.partial')
MANIFEST = CONFIG['fusion_manifest']
PROGRESS = OUTPUT.parent / 'fusion_progress.json'
GIB = 1024 ** 3
MIN_AVAILABLE = 8 * GIB
DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
          "I8": torch.int8, "I64": torch.int64, "U8": torch.uint8}

sys.path.insert(0, str(CORE))
import comfy.cli_args

# ComfyUI parses argv only when explicitly enabled; set its CPU option directly.
comfy.cli_args.args.cpu = True
import comfy.lora
import comfy.utils
from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
PROCESS = psutil.Process()
try:
    PROCESS.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except (psutil.AccessDenied, AttributeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        size = struct.unpack("<Q", prefix)[0]
        raw = stream.read(size)
    return json.loads(raw), 8 + size, prefix + raw


def no_cuda():
    assert torch.cuda.device_count() == 0, "CPU fusion must have no visible CUDA devices"
    assert not torch.cuda.is_initialized(), "CPU fusion must not initialize CUDA"


def wait_for_memory(stage):
    paused = False
    while psutil.virtual_memory().available < MIN_AVAILABLE:
        if not paused:
            print(json.dumps({"stage": stage, "paused_for_available_ram": True}), flush=True)
            paused = True
        write_json(PROGRESS, {"at": now(), "status": "waiting_for_available_ram",
                              "stage": stage, "available_gib": psutil.virtual_memory().available / GIB})
        time.sleep(2)


def tensor_read(stream, header, data_start, key):
    entry = header[key]
    begin, end = entry["data_offsets"]
    stream.seek(data_start + begin)
    raw = bytearray(stream.read(end - begin))
    assert len(raw) == end - begin
    return torch.frombuffer(raw, dtype=DTYPES[entry["dtype"]]).reshape(entry["shape"])


def tensor_write(stream, header, data_start, key, tensor):
    entry = header[key]
    assert tuple(tensor.shape) == tuple(entry["shape"]), key
    assert tensor.dtype == DTYPES[entry["dtype"]] and tensor.device.type == "cpu", key
    view = memoryview(tensor.detach().contiguous().view(torch.uint8).numpy()).cast("B")
    begin, end = entry["data_offsets"]
    assert view.nbytes == end - begin
    stream.seek(data_start + begin)
    stream.write(view)
    return hashlib.sha256(view).hexdigest()


def hash_region(stream, begin, size):
    stream.seek(begin)
    digest = hashlib.sha256()
    while size:
        chunk = stream.read(min(size, 8 * 1024 ** 2))
        assert chunk
        digest.update(chunk)
        size -= len(chunk)
    return digest.hexdigest()


def verify_file(path, manifest):
    source_header, source_start, _ = load_header(BASE)
    output_header, output_start, _ = load_header(path)
    source_keys = set(source_header) - {"__metadata__"}
    output_keys = set(output_header) - {"__metadata__"}
    assert source_keys == output_keys
    expected_changed = manifest["changed_tensor_sha256"]
    unchanged = 0
    int8_layers = 0
    with BASE.open("rb") as source, path.open("rb") as target:
        for key in sorted(source_keys, key=lambda k: source_header[k]["data_offsets"][0]):
            assert source_header[key] == output_header[key], key
            begin, end = source_header[key]["data_offsets"]
            actual = hash_region(target, output_start + begin, end - begin)
            if key in expected_changed:
                assert actual == expected_changed[key], key
            else:
                assert actual == hash_region(source, source_start + begin, end - begin), key
                unchanged += 1
            if key.endswith("comfy_quant"):
                conf = json.loads(bytes(tensor_read(target, output_header, output_start, key).tolist()))
                assert conf == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
                int8_layers += 1
    with safe_open(path, framework="pt", device="cpu") as opened:
        assert set(opened.keys()) == source_keys
        assert opened.metadata()["pdmd_lora_strength"] == "1.0"
        for record in manifest["modules"]:
            module = record["module"]
            assert opened.get_slice(module + ".weight").get_shape() == record["weight_shape"]
            if record["format"] == "int8_tensorwise_convrot":
                scale = opened.get_tensor(module + ".weight_scale")
                assert torch.isfinite(scale).all() and torch.all(scale > 0), module
    assert len(expected_changed) == 408 and int8_layers == 200
    assert len(manifest["modules"]) == 208
    assert sha256(BASE) == manifest["base"]["sha256"]
    assert sha256(LORA) == manifest["lora"]["sha256"]
    no_cuda()
    return {"verified_at": now(), "safetensors_readable": True, "tensor_count": len(source_keys),
            "changed_tensors_verified": len(expected_changed), "unchanged_tensors_bit_identical": unchanged,
            "int8_convrot_layers": int8_layers, "bf16_merged_layers": 8,
            "source_files_unchanged": True, "cuda_initialized": False}


def create():
    no_cuda()
    assert not OUTPUT.exists() and not PARTIAL.exists(), "Refuse to overwrite a final or partial model"
    assert not MANIFEST.exists(), "Existing fusion record; inspect before retrying"
    assert free_bytes(OUTPUT.parent) > 30 * 10 ** 9
    baseline_provenance = json.loads((ROOT / "model_recipe.json").read_text(encoding="utf-8"))
    lora_sha = sha256(LORA)
    assert lora_sha == baseline_provenance["lora"]["sha256"]
    header, data_start, original_header = load_header(BASE)
    for key, value in header.items():
        if key != "__metadata__":
            assert value["dtype"] in DTYPES, (key, value["dtype"])
    lora_header, lora_start, _ = load_header(LORA)
    prefixes = sorted(k.removesuffix(".lora_A.weight") for k in lora_header if k.endswith(".lora_A.weight"))
    assert len(prefixes) == 208
    assert set(lora_header) - {"__metadata__"} == {
        prefix + suffix for prefix in prefixes for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha")}
    for prefix in prefixes:
        assert prefix.startswith("diffusion_model.")
        module = prefix.removeprefix("diffusion_model.")
        shape = header[module + ".weight"]["shape"]
        assert lora_header[prefix + ".lora_B.weight"]["shape"][0] == shape[0]
        assert lora_header[prefix + ".lora_A.weight"]["shape"][1] == shape[1]

    metadata = dict(header.get("__metadata__", {}))
    metadata.update(pdmd_lora_sha256=lora_sha, pdmd_lora_strength="1.0", pdmd_lora_modules="208",
                    fusion_method="CPU FP32 ComfyUI LoRA; native INT8 ConvRot requantization; module-specific CRC seed",
                    fusion_created_at=now())
    header["__metadata__"] = metadata
    output_header = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    output_header += b" " * ((-len(output_header)) % 8)
    output_start = 8 + len(output_header)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    free_before = free_bytes(OUTPUT.parent)
    started = time.perf_counter()
    digest = hashlib.sha256(original_header)
    print(json.dumps({"stage": "copy_base_payload", "bytes": BASE.stat().st_size, "cpu_threads": 2}), flush=True)
    with BASE.open("rb") as source, PARTIAL.open("xb") as target:
        target.write(struct.pack("<Q", len(output_header)))
        target.write(output_header)
        source.seek(data_start)
        copied = 0
        while chunk := source.read(8 * 1024 ** 2):
            target.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
            if copied % (256 * 1024 ** 2) == 0:
                wait_for_memory("copy_base_payload")
                write_json(PROGRESS, {"at": now(), "stage": "copy_base_payload", "bytes_copied": copied})
        target.flush()
        os.fsync(target.fileno())

    source_paths = [CORE / "comfy/lora.py", CORE / "comfy/weight_adapter/lora.py",
                    CORE / "comfy/model_patcher.py", CORE / "comfy/ops.py", CORE / "comfy/utils.py",
                    Path(sys.modules[TensorWiseINT8Layout.__module__].__file__)]
    manifest = {"status": "fusion_in_progress", "started_at": now(), "cpu_threads": 2,
                "process_priority": str(PROCESS.nice()), "cuda_visible_devices": "-1", "cuda_initialized": False,
                "method": "Native ComfyUI CPU FP32 LoRA math, native INT8 ConvRot rowwise requantization, BF16 token-refiner output",
                "quality_limit": "CPU FP32 fusion and CPU requantization are not bit-identical to dynamic GPU LoRA",
                "base": {"path": str(BASE), "bytes": BASE.stat().st_size, "sha256": digest.hexdigest()},
                "lora": {"path": str(LORA), "bytes": LORA.stat().st_size, "sha256": lora_sha, "strength": 1.0},
                "output": {"path": str(OUTPUT)}, "script_sha256": sha256(Path(__file__)),
                "source_code_sha256": {str(p): sha256(p) for p in source_paths},
                "free_bytes_before": free_before, "changed_tensor_sha256": {}, "modules": []}
    write_json(MANIFEST, manifest)
    print(json.dumps({"stage": "merge_layers", "total_modules": 208, "cuda_initialized": False}), flush=True)

    with BASE.open("rb") as source, LORA.open("rb") as lora, PARTIAL.open("r+b") as target, torch.inference_mode():
        for index, prefix in enumerate(prefixes, 1):
            wait_for_memory("merge_layers")
            layer_start = time.perf_counter()
            module = prefix.removeprefix("diffusion_model.")
            key = module + ".weight"
            full_key = "diffusion_model." + key
            weight = tensor_read(source, header, data_start, key)
            tensors = {prefix + suffix: tensor_read(lora, lora_header, lora_start, prefix + suffix)
                       for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha")}
            rank = tensors[prefix + ".lora_A.weight"].shape[0]
            alpha = tensors[prefix + ".alpha"].item()
            assert alpha == rank and rank == (384 if ".qkv_proj" in prefix else 128)
            patch = comfy.lora.load_lora(tensors, {prefix: full_key})
            assert list(patch) == [full_key]
            quantized = module + ".comfy_quant" in header
            if quantized:
                conf = json.loads(bytes(tensor_read(source, header, data_start, module + ".comfy_quant").tolist()))
                assert conf == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
                scale = tensor_read(source, header, data_start, module + ".weight_scale")
                params = TensorWiseINT8Layout.Params(scale=scale, orig_dtype=torch.float32,
                    orig_shape=tuple(weight.shape), is_weight=True, convrot=True, convrot_groupsize=256)
                weight_float = QuantizedTensor(weight, "TensorWiseINT8Layout", params).dequantize()
            else:
                assert weight.dtype == torch.bfloat16
                weight_float = weight.float()
            rows = torch.cat((torch.arange(16), torch.arange(weight.shape[0] - 16, weight.shape[0])))
            before_sample = weight_float[rows].clone()
            merged = comfy.lora.calculate_weight([(1.0, patch[full_key], 1.0, None, None)],
                weight_float, full_key, intermediate_dtype=torch.float32)
            assert torch.isfinite(merged).all(), full_key
            reference = before_sample + tensors[prefix + ".lora_B.weight"][rows].float() @ tensors[prefix + ".lora_A.weight"].float()
            torch.testing.assert_close(merged[rows], reference, rtol=1e-4, atol=3e-5)
            record = {"module": module, "weight_shape": list(weight.shape), "rank": rank,
                      "alpha": alpha, "strength": 1.0, "sample_reference_max_abs_error": (merged[rows] - reference).abs().max().item()}
            if quantized:
                seed = comfy.utils.string_to_seed(full_key)
                q, new_params = TensorWiseINT8Layout.quantize(merged, scale="recalculate", stochastic_rounding=seed,
                    is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256)
                sample_params = dataclasses.replace(new_params, scale=new_params.scale[rows], orig_shape=tuple(q[rows].shape))
                restored = QuantizedTensor(q[rows], "TensorWiseINT8Layout", sample_params).dequantize()
                error = restored - merged[rows]
                record.update(format="int8_tensorwise_convrot", rounding_seed=seed,
                    sample_quantization_rmse=error.square().mean().sqrt().item(),
                    sample_quantization_relative_l2=(error.norm() / merged[rows].norm().clamp_min(1e-30)).item())
                manifest["changed_tensor_sha256"][key] = tensor_write(target, header, output_start, key, q)
                scale_key = module + ".weight_scale"
                manifest["changed_tensor_sha256"][scale_key] = tensor_write(target, header, output_start, scale_key, new_params.scale)
                del q, new_params, sample_params, restored, error, scale, params
            else:
                record["format"] = "bf16"
                manifest["changed_tensor_sha256"][key] = tensor_write(target, header, output_start, key, merged.to(torch.bfloat16))
            record.update(seconds=round(time.perf_counter() - layer_start, 3),
                          available_ram_gib=round(psutil.virtual_memory().available / GIB, 3))
            manifest["modules"].append(record)
            del weight, weight_float, merged, tensors, patch, before_sample, reference, rows
            gc.collect()
            no_cuda()
            write_json(MANIFEST, manifest)
            write_json(PROGRESS, {"at": now(), "stage": "merge_layers", "modules_completed": index,
                                  "modules_total": 208, "module": module,
                                  "process_rss_gib": round(PROCESS.memory_info().rss / GIB, 3),
                                  "available_ram_gib": record["available_ram_gib"], "cuda_initialized": False})
            if index % 10 == 0 or index == 208:
                print(json.dumps({"merged": index, "total": 208, "seconds": round(time.perf_counter() - started, 1)}), flush=True)
        target.flush()
        os.fsync(target.fileno())

    print(json.dumps({"stage": "verify_full_file", "changed_tensors": len(manifest["changed_tensor_sha256"])}), flush=True)
    manifest["verification"] = verify_file(PARTIAL, manifest)
    manifest["output"].update(bytes=PARTIAL.stat().st_size, sha256=sha256(PARTIAL))
    # Same-volume publication, without a second model copy; Windows rename refuses overwrite.
    PARTIAL.rename(OUTPUT)
    manifest.update(status="created_and_cpu_verified", finished_at=now(),
        seconds=round(time.perf_counter() - started, 3),
        peak_process_ram_gib=round(PROCESS.memory_info().peak_wset / GIB, 3),
        free_bytes_after=free_bytes(OUTPUT.parent), generation_status="not_started")
    write_json(MANIFEST, manifest)
    write_json(PROGRESS, {"at": now(), "stage": "complete", "modules_completed": 208, "cuda_initialized": False})
    print(json.dumps({"completed": True, "bytes": OUTPUT.stat().st_size, "seconds": manifest["seconds"],
                      "peak_process_ram_gib": manifest["peak_process_ram_gib"], "cuda_initialized": False}), flush=True)


if __name__ == "__main__":
    if options.verify:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert sha256(OUTPUT) == data["output"]["sha256"]
        print(json.dumps(verify_file(OUTPUT, data), ensure_ascii=True), flush=True)
    else:
        create()
