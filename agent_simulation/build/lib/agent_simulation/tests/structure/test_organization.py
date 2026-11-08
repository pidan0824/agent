"""Keep data loading and dataset preparation independent of runtime services."""

from agent_simulation import PROJECT_ROOT

import ast
import sys
import unittest
import subprocess

ROOT = PROJECT_ROOT
SHARED_BATCHING = "agent_simulation.environment.batching"
RUNTIME_MODULES = (
    "agent_simulation.environment",
    "agent_simulation.agents.base",
    "agent_simulation.inference.local",
    "agent_simulation.logs",
    "agent_simulation.simulation",
    "agent_simulation.agents.decisions",
    "agent_simulation.agents.memory.reflection",
    "agent_simulation.agents.customer",
    "agent_simulation.agents.seller",
)


class OrganizationTests(unittest.TestCase):
    def test_sft_loader_does_not_import_runtime(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from agent_simulation.datasets.labels import SFTDataset; "
                f"assert not any(m.startswith({RUNTIME_MODULES!r}) for m in sys.modules "
                f"if m not in ('agent_simulation.environment', {SHARED_BATCHING!r}))",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dataset_preparation_does_not_import_training(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import agent_simulation.datasets.tokens; "
             "assert not any(m.startswith('agent_simulation.training') for m in sys.modules)"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for path in (ROOT / 'agent_simulation/datasets').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                           else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                self.assertFalse(any(m.startswith('agent_simulation.training') for m in modules), str(path))

    def test_data_package_has_no_agent_dependencies(self):
        for path in (ROOT / "agent_simulation/data").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [a.name for a in node.names] if isinstance(node, ast.Import) else []
                )
                for name in modules:
                    belongs_to_data = name == "agent_simulation.data" or name.startswith(
                        "agent_simulation.data."
                    )
                    self.assertFalse(name.startswith("agent_simulation.") and not belongs_to_data, str(path))
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from agent_simulation.data.catalog import Catalog; "
                "assert not any(n.startswith('agent_simulation.') and n != 'agent_simulation.data' "
                "and not n.startswith('agent_simulation.data.') for n in sys.modules)",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
