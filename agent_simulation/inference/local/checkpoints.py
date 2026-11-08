"""Verify local Qwen checkpoints and attach verified LoRA weights."""

import hashlib
import json
from pathlib import Path

from agent_simulation.data.common import require, sha256


def attach_adapter(model, directory, *, expected_role=None, name="default", checkpoint=None):
    """Mount a verified local adapter for offline evaluation or routed Episode inference."""
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    directory = Path(directory)
    existing = isinstance(model.model, PeftModel)
    require(not existing or (name != "default" and name not in model.model.peft_config), "A LoRA adapter is already attached")
    if expected_role is not None:
        require(expected_role in {"customer", "seller"}, "Unknown adapter role")
        metadata = directory.parent / "run_config.json"
        require(metadata.is_file(), "Cannot verify adapter role: missing training run_config.json")
        training = json.loads(metadata.read_text())["identity"]
        require(training.get("role") == expected_role,
                "Adapter training role differs from the requested Agent")
        if checkpoint is not None:
            files = checkpoint["files"]
            shards = {p: entry["sha256"] for p, entry in files.items() if p.endswith(".safetensors")}
            require(training["base_config_sha256"] == files["config.json"]["sha256"]
                    and training["base_weight_sha256"] == shards, "Adapter training base differs")
    config = directory / "adapter_config.json"
    weights = directory / "adapter_model.safetensors"
    require(config.is_file() and weights.is_file(), "Missing local LoRA adapter files")
    require(json.loads(config.read_text())["peft_type"] == "LORA", "Expected a LoRA adapter")
    identity = {p.name: sha256(p) for p in (config, weights)}
    saved = load_file(str(weights), device="cpu")
    require(saved and all(t.is_floating_point() and torch.isfinite(t).all().item()
                          for t in saved.values()), "Nonfinite or invalid LoRA weights")
    if existing:
        loaded = model.model
        loaded.load_adapter(str(directory), adapter_name=name, is_trainable=False, local_files_only=True)
    else:
        loaded = PeftModel.from_pretrained(model.model, str(directory), adapter_name=name,
                                           is_trainable=False, local_files_only=True)
    try:
        actual = get_peft_model_state_dict(loaded, adapter_name=name, save_embedding_layers=False)
        # PEFT can load incomplete checkpoints with a warning.
        require(actual.keys() == saved.keys(), "LoRA checkpoint keys do not match the loaded adapter")
        loaded.eval()
        loaded.requires_grad_(False)
    except Exception:
        if existing:
            loaded.delete_adapter(name)
        else:
            loaded.unload()
        raise
    model.model = loaded
    model.generator_id += ":adapter=" + str(directory) + ":" + identity[weights.name]
    return identity


def verify_checkpoint(directory, revision):
    directory = Path(directory)
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    files = sorted(
        set(index["weight_map"].values())
        | {
            "model.safetensors.index.json",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        }
    )
    result = {}
    for name in files:
        require(Path(name).name == name, "Unexpected checkpoint file path")
        path = directory / name
        metadata = directory / ".cache/huggingface/download" / (name + ".metadata")
        commit, etag, *_ = metadata.read_text().splitlines()
        require(commit == revision, "Checkpoint file is from a different revision: " + name)
        size = path.stat().st_size
        sha = hashlib.sha256()
        blob = None if len(etag) == 64 else hashlib.sha1(b"blob " + str(size).encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
                if blob is not None:
                    blob.update(chunk)
        digest = sha.hexdigest()
        require(
            etag == (digest if blob is None else blob.hexdigest()),
            "Checkpoint checksum mismatch: " + name,
        )
        result[name] = {"bytes": size, "sha256": digest, "etag": etag}
    return {"revision": revision, "files": result, "verified": True}
