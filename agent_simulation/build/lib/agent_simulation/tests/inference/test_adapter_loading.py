"""Check exact local LoRA restoration and reject incomplete or invalid weights."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import warnings

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from agent_simulation.inference.local.checkpoints import attach_adapter
from agent_simulation.tests.fixtures import tiny_base


class AdapterLoadingTests(unittest.TestCase):
    def test_role_batch_matches_each_single_adapter_and_unadapted_memory(self):
        from agent_simulation.inference.local.qwen import QwenLocal
        with TemporaryDirectory() as folder:
            backend = object.__new__(QwenLocal)
            backend.generator_id = "fixture"
            backend.model = tiny_base().eval()
            base = deepcopy(backend.model)
            tokens = torch.tensor([[1, 2, 3]])
            expected = []
            checkpoint = {"files": {"config.json": {"sha256": "config"},
                                    "weights.safetensors": {"sha256": "weights"}}}
            backend.role_adapters = ("customer", "seller")
            for role in backend.role_adapters:
                path = Path(folder) / role / "adapter"
                _, source, _ = self.fixture(path)
                if role == "seller":
                    with torch.no_grad():
                        for n, parameter in source.named_parameters():
                            if "lora_B" in n:
                                parameter.mul_(-1)
                    source.save_pretrained(path)
                (path.parent / "run_config.json").write_text(json.dumps({"identity": {
                    "role": role, "base_config_sha256": "config",
                    "base_weight_sha256": {"weights.safetensors": "weights"}}}))
                attach_adapter(backend, path, expected_role=role, name=role, checkpoint=checkpoint)
                expected.append(source(tokens).logits)
            kwargs = backend._adapter_args([{"agent_role": "customer"}, {"agent_role": "seller"}, {}])
            logits = backend.model(tokens.repeat(3, 1), **kwargs).logits
            self.assertFalse(torch.allclose(expected[0], expected[1]))
            torch.testing.assert_close(logits, torch.cat([*expected, base(tokens).logits]), atol=1e-6, rtol=1e-5)
            generated = backend.model.generate(tokens.repeat(3, 1), max_new_tokens=1, do_sample=False, **kwargs)
            torch.testing.assert_close(generated[:, -1], logits[:, -1].argmax(-1))

    def fixture(self, directory):
        base = tiny_base().eval()
        source = get_peft_model(
            deepcopy(base),
            LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                       target_modules=["q_proj", "v_proj"], lora_dropout=0),
        )
        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(mean=0, std=0.1)
        source.eval()
        source.save_pretrained(directory)
        return base, source, SimpleNamespace(model=base, generator_id="tiny-fixture")

    def test_loaded_adapter_changes_logits_and_disabled_adapter_recovers_base(self):
        with TemporaryDirectory() as directory:
            base, source, backend = self.fixture(directory)
            tokens = torch.tensor([[1, 3, 4, 5]])
            with torch.no_grad():
                baseline = base(tokens).logits
                expected = source(tokens).logits
            self.assertFalse(torch.equal(baseline, expected))
            identity = attach_adapter(backend, Path(directory))
            self.assertEqual(set(identity), {"adapter_config.json", "adapter_model.safetensors"})
            self.assertIn("adapter=", backend.generator_id)
            self.assertFalse(backend.model.training)
            self.assertTrue(all(not p.requires_grad for p in backend.model.parameters()))
            with torch.no_grad():
                enabled = backend.model(tokens).logits
                with backend.model.disable_adapter():
                    disabled = backend.model(tokens).logits
                restored = backend.model(tokens).logits
            torch.testing.assert_close(enabled, expected)
            torch.testing.assert_close(disabled, baseline)
            torch.testing.assert_close(restored, enabled)
            self.assertEqual(backend.model.active_adapters, ["default"])

    def test_missing_extra_nonfinite_and_misshaped_tensors_are_rejected(self):
        for corruption in ("missing", "extra", "nan", "inf", "shape"):
            with self.subTest(corruption=corruption), TemporaryDirectory() as directory:
                _, _, backend = self.fixture(directory)
                path = Path(directory) / "adapter_model.safetensors"
                weights = load_file(str(path))
                first = next(iter(weights))
                if corruption == "missing":
                    weights.pop(first)
                elif corruption == "extra":
                    weights["unexpected.lora_A.weight"] = torch.ones(2, 8)
                elif corruption == "shape":
                    weights[first] = weights[first][:, :-1].contiguous()
                else:
                    weights[first].view(-1)[0] = float(corruption)
                save_file(weights, str(path), metadata={"format": "pt"})
                with warnings.catch_warnings(record=True):
                    with self.assertRaises((ValueError, RuntimeError)):
                        attach_adapter(backend, Path(directory))
                self.assertEqual(backend.generator_id, "tiny-fixture")

    def test_attaching_twice_cannot_silently_stack_adapters(self):
        with TemporaryDirectory() as directory:
            _, _, backend = self.fixture(directory)
            attach_adapter(backend, Path(directory))
            loaded = backend.model
            with self.assertRaisesRegex(ValueError, "already attached"):
                attach_adapter(backend, Path(directory))
            self.assertIs(backend.model, loaded)

    def test_role_mismatch_and_missing_role_metadata_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            _, _, backend = self.fixture(adapter)
            with self.assertRaisesRegex(ValueError, "missing training run_config"):
                attach_adapter(backend, adapter, expected_role="seller")
            (root / "run_config.json").write_text(json.dumps({"identity": {"role": "customer"}}))
            with self.assertRaisesRegex(ValueError, "role differs"):
                attach_adapter(backend, adapter, expected_role="seller")
            self.assertEqual(backend.generator_id, "tiny-fixture")
            attach_adapter(backend, adapter, expected_role="customer")
            self.assertEqual(backend.model.active_adapters, ["default"])


if __name__ == "__main__":
    unittest.main()
