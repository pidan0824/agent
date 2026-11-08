"""Test checkpoint revision and file integrity."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from agent_simulation.inference.local.checkpoints import verify_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_bytes_and_revision_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = root / ".cache/huggingface/download"
            meta.mkdir(parents=True)
            names = ["config.json", "generation_config.json", "tokenizer.json",
                     "tokenizer_config.json", "vocab.json", "merges.txt", "weights.safetensors",
                     "model.safetensors.index.json"]
            for name in names:
                content = (json.dumps({"weight_map": {"a": "weights.safetensors"}}).encode()
                           if name.endswith("index.json") else b"test fixture")
                (root / name).write_bytes(content)
                etag = (hashlib.sha256(content).hexdigest() if name.endswith("safetensors")
                        else hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest())
                (meta / (name + ".metadata")).write_text("revision\n" + etag + "\n0\n")
            self.assertTrue(verify_checkpoint(root, "revision")["verified"])
            with self.assertRaisesRegex(ValueError, "different revision"):
                verify_checkpoint(root, "other")
            (root / "weights.safetensors").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_checkpoint(root, "revision")
