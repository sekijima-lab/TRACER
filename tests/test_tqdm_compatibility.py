"""Verify progress iteration and the patched CLI argument security boundary."""
import io
import os
import subprocess
import sys
import unittest

from tqdm import tqdm
from tqdm.auto import tqdm as auto_tqdm


class TqdmCompatibility(unittest.TestCase):
    def test_smiles_order_and_values(self):
        samples = ["CCO", "c1ccccc1", "CC(=O)O", "C[C@H](N)C(=O)O", "C1CCCCC1"]
        for progress in (tqdm, auto_tqdm):
            output = io.StringIO()
            with progress(samples, file=output, mininterval=0) as bar:
                actual = [smiles for smiles in bar]
                self.assertEqual(bar.n, len(samples))
                self.assertEqual(bar.total, len(samples))
            self.assertEqual(actual, samples)
            self.assertIn("5/5", output.getvalue())
            self.assertEqual(list(progress([], disable=True)), [])

    def test_file_pipeline_preserves_input(self):
        data = "CCO\nc1ccccc1\nCC(=O)O\n"
        process = subprocess.run([sys.executable, "-m", "tqdm", "--total", "3"], input=data, text=True, capture_output=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, data)
        self.assertIn("3/3", process.stderr)

    @unittest.skipIf(os.environ.get("TQDM_BASELINE") == "1", "old CLI evaluates input")
    def test_cli_rejects_python_expression(self):
        # A harmless expression only: no imports, filesystem writes, or commands.
        process = subprocess.run([sys.executable, "-m", "tqdm", "--buf-size", '1" + str(1+1) + "'], input="CCO\n", text=True, capture_output=True, timeout=15)
        self.assertNotEqual(process.returncode, 0, "CLI must not evaluate a Python expression")


if __name__ == "__main__":
    unittest.main(verbosity=2)
