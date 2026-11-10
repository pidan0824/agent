"""Compare fixed Lite records with the reviewed SFT outputs."""

from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import unittest

from agent_simulation import PROJECT_ROOT
from agent_simulation.datasets.labels import SFTDataset
from agent_simulation.datasets.labels import record_samples
from agent_simulation.tests.fixtures import sample_identity
from agent_simulation.datasets.tokenization import encode_tokens


# Complete model payloads, then encoded tokens, answer boundaries and weights.
EXPECTED = {
    ("customer", 14): (
        "efea01a64dcfe673a4ae0c6195dd25cd0c270f0884a6a5ed0454fea3e6ab9b93",
        "984c31a5288938bba093a7d653e59d09392f7fc5718045e1be84b6cf67aae161",
    ),
    ("customer", 12758): (
        "0bdf4c525df10808af37fbad7d6544b1b93fccf21ab04ea938c8b37064e442e0",
        "1db7e3a30bba35012d0259e9862a2021abb812e01555e9a646352c8f973a8564",
    ),
    ("seller", "seller:108629:579227:850769"): (
        "d0dde58b0850f4e084d4a031ca1fa78f6d2c914989f07bddf8258b56d4c6886e",
        "821066366bd8af90dcf81ad88ee328b42d6164cc8356d14fec8b2f95b2a60387",
    ),
}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SFTRegressionTests(unittest.TestCase):
    def test_reviewed_lite_sample_outputs(self):
        directory = PROJECT_ROOT / "data/kuaisearch/lite/sft"
        tokenizer = PROJECT_ROOT / "models/Qwen2.5-3B-Instruct"
        manifest = json.loads((directory / "manifest.json").read_text())
        catalog = Path(manifest["catalog"])
        if not (catalog / "catalog.duckdb").is_file() or not (tokenizer / "tokenizer.json").is_file():
            self.skipTest("The fixed Lite catalog and tokenizer are required")

        with closing(SFTDataset(directory, tokenizer)) as data:
            for (role, record), expected in EXPECTED.items():
                with self.subTest(role=role, record=record):
                    samples = record_samples(data, role, record)
                    actual = (
                        digest([sample_identity(sample) for sample in samples]),
                        digest([encode_tokens(sample, data.tokenizer) for sample in samples]),
                    )
                    self.assertEqual(
                        actual, expected,
                        "SFT output changed. If intentional, review the samples and update the "
                        "baseline. Build a fresh Token cache before training on the updated inputs.",
                    )
