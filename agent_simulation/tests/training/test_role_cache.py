"""A complete role can train while another role is absent or unfinished."""
import importlib.metadata
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from agent_simulation.datasets.tokens import PreparedRow, RecordWriter
from agent_simulation.training.run import check_cached_data, parser, train, training_identity
from agent_simulation.datasets.tokens import selected_identity, validate_release
from agent_simulation.data.common import sha256


class RoleCacheTests(unittest.TestCase):
    def fixture(self, root, role):
        dataset = root / 'dataset'
        dataset.mkdir()
        (dataset / 'manifest.json').write_text(json.dumps({'counts': {
            r: {'train': 1, 'test': 1} for r in ('customer', 'seller')}}))
        selection = dict(schema='sft-split-candidates-v2', seed=42,
                         dataset_manifest_sha256=sha256(dataset / 'manifest.json'),
                         roles={r: {'train': {}, 'test': {}} for r in ('customer', 'seller')})
        manifest = root / 'candidate_manifest.json'
        manifest.write_text(json.dumps(selection))
        spec = selected_identity(dataset, selection, sha256(manifest), 42, 2)
        (root / 'identity.json').write_text(json.dumps(spec))
        args = SimpleNamespace(output=root, reserve_gib=0)
        for split in ('train', 'test'):
            for bucket in range(2):
                output = RecordWriter(args, role, split, bucket, expected=1)
                output.add(str(bucket), [PreparedRow(f'{role}:{bucket}',
                           np.array([1, 2, 3], dtype=np.int32), 2, 1.)])
                output.finish()
        args = SimpleNamespace(token_cache=root, dataset=dataset, seed=42,
                               validation_samples=128, role=role)
        data = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        return args, data, spec

    def test_stage_filter_preserves_response_weights_and_epoch_order(self):
        from agent_simulation.training.data_loader import TokenStream
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "identity.json").write_text(json.dumps({"buckets": 1}))
            args = SimpleNamespace(output=root, reserve_gib=0)
            writer = RecordWriter(args, "customer", "train", 0, expected=2)
            for request in (1, 2):
                writer.add(str(request), [
                    PreparedRow(f"{stage}:{request}:{batch}", np.array([1, 2, 3], dtype=np.int32), 2, weight)
                    for stage, batch, weight in (("query", 0, 1.), ("response", 1, .5), ("response", 2, .5))
                ])
            writer.finish()
            query = TokenStream(root, "customer", "train", stage="query")
            response = TokenStream(root, "customer", "train", stage="response", shuffle_buffer=3)
            self.assertEqual(len(query), 2)
            self.assertEqual(len(response), 4)
            self.assertEqual(sum(r.weight for r in response), 2.)
            response.set_epoch(2)
            first = [r.sample_id for r in response]
            response.set_epoch(2)
            self.assertEqual(first, [r.sample_id for r in response])
            self.assertTrue(all(s.startswith("response:") for s in first))

    def test_each_complete_role_passes_both_training_entry_checks(self):
        for role in ('customer', 'seller'):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args, data, spec = self.fixture(root, role)
                other = 'seller' if role == 'customer' else 'customer'
                (root / other / 'train' / '000').mkdir(parents=True)
                checked = check_cached_data(args, data, [])
                self.assertEqual(checked['counts'], {'train': 2, 'test': 2})
                self.assertEqual(checked['token_cache_identity'], spec)
                self.assertEqual(checked['checks'], [])
                self.assertEqual(len(check_cached_data(args, data, [], verify=True)['checks']), 4)
                self.assertFalse((root / 'complete.json').exists())
                with self.assertRaises(FileNotFoundError):
                    validate_release(root, args.dataset, 42)
                cli = parser().parse_args(['train', '--model', 'unused', '--tokenizer', 'unused',
                    '--dataset', str(args.dataset), '--token-cache', str(root), '--role', role,
                    '--output', str(root / 'training'), '--device', 'cpu'])
                # Exercise the production entry up to weight loading, without a model or SFT.
                with patch('agent_simulation.training.run.check_data',
                           side_effect=lambda a: check_cached_data(a, data, [])), \
                     patch('agent_simulation.training.run.training_identity', return_value={}), \
                     patch('agent_simulation.training.run.load_tokenizer', return_value=data.tokenizer), \
                     patch('agent_simulation.training.run.AutoConfig.from_pretrained',
                           return_value=SimpleNamespace(model_type="qwen2")), \
                     patch('agent_simulation.training.run.AutoModelForCausalLM.from_pretrained',
                           side_effect=RuntimeError('reached weight loading')) as load:
                    with self.assertRaisesRegex(RuntimeError, 'reached weight loading'):
                        train(cli)
                    load.assert_called_once()
                self.assertFalse((root / 'complete.json').exists())


    def test_partial_role_and_wrong_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, data, spec = self.fixture(root, 'customer')
            with self.assertRaisesRegex(ValueError, 'seed differs'):
                validate_release(root, args.dataset, 43, role='customer')
            marker = root / 'customer' / 'test' / '001' / 'complete.json'
            marker.unlink()
            with self.assertRaisesRegex(ValueError, 'unfinished buckets'):
                check_cached_data(args, data, [])
            (root / 'complete.json').write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'unfinished buckets'):
                check_cached_data(args, data, [])

    def test_resume_rejects_changed_cache_manifests_and_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, data, _ = self.fixture(root, 'seller')
            (root / 'config.json').write_text('{}')
            run = root / 'training'
            run.mkdir()
            cli = parser().parse_args(['train', '--model', str(root), '--dataset', str(args.dataset),
                                       '--role', 'seller', '--token-cache', str(root), '--output', str(run),
                                       '--device', 'cpu', '--dtype', 'float32',
                                       '--tokenizer', str(root), '--resume', 'latest'])
            checked = check_cached_data(args, data, [])
            bound = training_identity(cli, checked, 'test')
            self.assertNotIn("training_code_sha256", bound)
            packages = {name: importlib.metadata.version(name) for name in
                        ('torch', 'transformers', 'peft', 'accelerate', 'duckdb', 'numpy', 'tokenizers')}
            config = run / 'run_config.json'
            marker = root / 'seller/train/000/complete.json'
            original = marker.read_text()
            manifest = json.loads(original)
            manifest['files'][0]['sha256'] = '0' * 64
            marker.write_text(json.dumps(manifest))
            with patch('agent_simulation.training.run.check_data', return_value=checked), \
                 patch('agent_simulation.training.run.resume_checkpoint', return_value=run / 'checkpoint-1'), \
                 patch('agent_simulation.training.run.AutoConfig.from_pretrained',
                       return_value=SimpleNamespace(model_type="qwen2")), \
                 patch('agent_simulation.training.run.AutoModelForCausalLM.from_pretrained') as load:
                config.write_text(json.dumps(dict(identity=bound, packages=packages)))
                with self.assertRaisesRegex(ValueError, 'Resume data/model/config differs'):
                    train(cli)
                marker.write_text(original)
                config.write_text(json.dumps(dict(identity=bound, packages={**packages, 'torch': 'changed'})))
                with self.assertRaisesRegex(ValueError, 'Resume package versions differ'):
                    train(cli)
                load.assert_not_called()
                # Old metadata can include the retired full-Train evaluation flag.
                saved = {**bound, "training_code_sha256": {"run.py": "old source hash"},
                         "evaluate_train": True}
                config.write_text(json.dumps(dict(identity=saved, packages=packages)))
                load.side_effect = RuntimeError("reached weight loading")
                with patch('agent_simulation.training.run.load_tokenizer', return_value=data.tokenizer):
                    with self.assertRaisesRegex(RuntimeError, 'reached weight loading'):
                        train(cli)
                load.assert_called_once()

    def test_warm_start_validates_role_and_keeps_adapter_rank_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, data, _ = self.fixture(root, "seller")
            (root / "config.json").write_text("{}")
            source = root / "source"
            adapter = source / "adapter"
            adapter.mkdir(parents=True)
            adapter_config = json.dumps({"peft_type": "LORA", "r": 2})
            (adapter / "adapter_config.json").write_text(adapter_config)
            output = root / "training"
            cli = ["train", "--model", str(root), "--dataset", str(args.dataset),
                   "--role", "seller", "--token-cache", str(root), "--output", str(output), "--device", "cpu"]
            warm = parser().parse_args([*cli, "--init-adapter", str(adapter)])
            metadata = source / "run_config.json"
            metadata.write_text(json.dumps({"identity": {"role": "customer"}}))
            with self.assertRaisesRegex(ValueError, "Adapter training role differs"):
                train(warm)

            metadata.write_text(json.dumps({"identity": {
                "role": "seller", "base_config_sha256": sha256(root / "config.json"),
                "base_weight_sha256": {},
            }}))
            checked = check_cached_data(args, data, [])
            with patch("agent_simulation.training.run.check_data", return_value=checked), \
                 patch("agent_simulation.training.run.load_tokenizer", return_value=data.tokenizer), \
                 patch("agent_simulation.training.run.load_base_model"), \
                 patch("agent_simulation.training.run.PeftModel.from_pretrained",
                       side_effect=RuntimeError("reached adapter loading")) as load:
                with self.assertRaisesRegex(RuntimeError, "reached adapter loading"):
                    train(warm)
                self.assertEqual(load.call_args.args[1], adapter)
                identity = json.loads((output / "run_config.json").read_text())["identity"]
                self.assertEqual(identity["rank"], 2)
                self.assertEqual(identity["init_adapter"], str(adapter.resolve()))

                metadata.unlink()
                checkpoint = output / "checkpoint-1"
                checkpoint.mkdir()
                (checkpoint / "adapter_config.json").write_text(adapter_config)
                resume = parser().parse_args([*cli, "--resume", "latest"])
                with patch("agent_simulation.training.run.resume_checkpoint", return_value=checkpoint):
                    with self.assertRaisesRegex(RuntimeError, "reached adapter loading"):
                        train(resume)
                self.assertEqual(load.call_args.args[1], checkpoint)

    def test_global_identity_is_required_and_shard_checksums_are_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, data, spec = self.fixture(root, 'customer')
            (root / 'complete.json').write_text(json.dumps({**spec, 'seed': 99}))
            with self.assertRaisesRegex(ValueError, 'completion identity'):
                check_cached_data(args, data, [])
            (root / 'complete.json').unlink()
            shard = root / 'customer' / 'train' / '000' / '000000.npz'
            shard.write_bytes(b'corrupt')
            self.assertEqual(check_cached_data(args, data, [])['checks'], [])
            with self.assertRaisesRegex(ValueError, 'Changed token shard'):
                check_cached_data(args, data, [], verify=True)
