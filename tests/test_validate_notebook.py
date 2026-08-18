import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_notebook.py"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def code(
    source: str,
    *,
    outputs: Optional[List[dict]] = None,
    execution_count=None,
) -> dict:
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs or [],
        "source": [source],
    }


def valid_notebook() -> dict:
    return {
        "cells": [
            markdown("# Approval Agent\n\n## Overview\nA safe agent."),
            markdown("## Detailed Explanation\n\n### Agent Architecture\nThe workflow."),
            markdown("## Required Packages\n\n### Install dependencies"),
            code("print('ready')"),
            markdown("## Implementation\n\n### Define the policy"),
            code("risk = 'low'"),
            markdown("## Usage Example\nRun an example."),
            markdown("## Comparison\nCompare approaches."),
            markdown("## Additional Considerations\nLimitations."),
            markdown("## References\n- Example"),
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class NotebookValidatorTests(unittest.TestCase):
    def run_validator(self, *paths: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(path) for path in paths)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def write_notebook(self, directory: Path, data: dict, name: str = "agent.ipynb") -> Path:
        path = directory / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_accepts_template_compliant_notebook(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_notebook(Path(temp), valid_notebook())
            result = self.run_validator(path)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_reports_outputs_and_execution_counts(self):
        notebook = valid_notebook()
        notebook["cells"][3]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["ready\n"]}]
        notebook["cells"][3]["execution_count"] = 1

        with tempfile.TemporaryDirectory() as temp:
            path = self.write_notebook(Path(temp), notebook)
            result = self.run_validator(path)

        self.assertEqual(result.returncode, 1)
        self.assertIn("NB002", result.stdout)

    def test_reports_code_cell_without_preceding_markdown_description(self):
        notebook = valid_notebook()
        notebook["cells"].insert(4, code("print('undocumented')"))

        with tempfile.TemporaryDirectory() as temp:
            path = self.write_notebook(Path(temp), notebook)
            result = self.run_validator(path)

        self.assertEqual(result.returncode, 1)
        self.assertIn("NB003", result.stdout)
        self.assertIn("cell 5", result.stdout)

    def test_reports_code_cell_after_empty_markdown(self):
        notebook = valid_notebook()
        notebook["cells"].insert(4, markdown("   \n"))
        notebook["cells"].insert(5, code("print('undocumented')"))

        with tempfile.TemporaryDirectory() as temp:
            path = self.write_notebook(Path(temp), notebook)
            result = self.run_validator(path)

        self.assertEqual(result.returncode, 1)
        self.assertIn("NB003", result.stdout)
        self.assertIn("cell 6", result.stdout)

    def test_reports_missing_template_section(self):
        notebook = valid_notebook()
        notebook["cells"] = [
            cell for cell in notebook["cells"]
            if "## Comparison" not in "".join(cell.get("source", []))
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = self.write_notebook(Path(temp), notebook)
            result = self.run_validator(path)

        self.assertEqual(result.returncode, 1)
        self.assertIn("NB004", result.stdout)
        self.assertIn("Comparison", result.stdout)

    def test_reports_missing_local_markdown_image(self):
        notebook = valid_notebook()
        notebook["cells"][1]["source"] = [
            "## Detailed Explanation\n\n![Architecture](../images/missing.svg)"
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = self.write_notebook(Path(temp), notebook)
            result = self.run_validator(path)

        self.assertEqual(result.returncode, 1)
        self.assertIn("NB005", result.stdout)
        self.assertIn("missing.svg", result.stdout)

    def test_reports_missing_local_html_image(self):
        notebook = valid_notebook()
        notebook["cells"][1]["source"] = [
            '## Detailed Explanation\n\n<img src="../images/missing.png" alt="Architecture">'
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = self.write_notebook(Path(temp), notebook)
            result = self.run_validator(path)

        self.assertEqual(result.returncode, 1)
        self.assertIn("NB005", result.stdout)
        self.assertIn("missing.png", result.stdout)

    def test_malformed_root_reports_nb001_and_does_not_abort_later_files(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            malformed = directory / "malformed.ipynb"
            malformed.write_text("[]", encoding="utf-8")
            good = self.write_notebook(directory, valid_notebook(), "good.ipynb")
            result = self.run_validator(malformed, good)

        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn(f"FAIL {malformed}", result.stdout)
        self.assertIn("NB001", result.stdout)
        self.assertIn(f"PASS {good}", result.stdout)

    def test_malformed_cell_source_reports_nb001_and_continues(self):
        malformed_data = valid_notebook()
        malformed_data["cells"][0]["source"] = [1]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            malformed = self.write_notebook(directory, malformed_data, "malformed.ipynb")
            good = self.write_notebook(directory, valid_notebook(), "good.ipynb")
            result = self.run_validator(malformed, good)

        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn(f"FAIL {malformed}", result.stdout)
        self.assertIn("NB001", result.stdout)
        self.assertIn(f"PASS {good}", result.stdout)

    def test_cli_checks_every_requested_notebook(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            good = self.write_notebook(directory, valid_notebook(), "good.ipynb")
            bad_data = valid_notebook()
            bad_data["cells"][3]["execution_count"] = 2
            bad = self.write_notebook(directory, bad_data, "bad.ipynb")
            result = self.run_validator(good, bad)

        self.assertEqual(result.returncode, 1)
        self.assertIn(f"PASS {good}", result.stdout)
        self.assertIn(f"FAIL {bad}", result.stdout)


if __name__ == "__main__":
    unittest.main()
