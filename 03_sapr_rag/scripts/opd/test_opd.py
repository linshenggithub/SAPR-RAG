#!/usr/bin/env python3
"""Unit tests for SAPR-OPD data and reward plumbing."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


OPD_DIR = Path(__file__).resolve().parent
GRPO_DIR = OPD_DIR.parent / "grpo"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_opd_dataset = load_module("build_opd_dataset", OPD_DIR / "build_opd_dataset.py")


class TestBuildOPDDataset(unittest.TestCase):

    def test_clean_row_removes_all_privileged_fields(self):
        row = {
            "messages": [{"role": "user", "content": "Question: q"}],
            "golden_answers": ["a"],
            "gold_titles": ["t"],
            "gold_sup_sents": ["s"],
            "source": "hotpotqa",
            "teacher_prompt": "gold",
            "teacher_query_prompt": "query plan",
            "teacher_answer_prompt": "answer",
        }
        cleaned = build_opd_dataset.clean_row(row, 1)
        self.assertEqual(cleaned["golden_answers"], ["a"])
        self.assertFalse(any(key.startswith("teacher_") for key in cleaned))

    def test_clean_row_requires_reward_fields(self):
        with self.assertRaisesRegex(ValueError, "golden_answers"):
            build_opd_dataset.clean_row({"messages": [], "source": "hotpotqa"}, 3)

    def test_balanced_reservoir_sample_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_a = Path(tmpdir) / "a.jsonl"
            output_b = Path(tmpdir) / "b.jsonl"
            with input_path.open("w", encoding="utf-8") as writer:
                for source in ("hotpotqa", "2wiki", "musique"):
                    for index in range(10):
                        row = {
                            "messages": [{"role": "user", "content": f"{source}-{index}"}],
                            "golden_answers": ["a"],
                            "gold_titles": ["t"],
                            "gold_sup_sents": ["s"],
                            "source": source,
                            "teacher_prompt": "must disappear",
                        }
                        writer.write(json.dumps(row) + "\n")

            counts_a = build_opd_dataset.build_dataset(input_path, output_a, 3, seed=7)
            counts_b = build_opd_dataset.build_dataset(input_path, output_b, 3, seed=7)
            self.assertEqual(counts_a, {"hotpotqa": 3, "2wiki": 3, "musique": 3})
            self.assertEqual(output_a.read_text(), output_b.read_text())
            self.assertNotIn("teacher_prompt", output_a.read_text())


class TestSaprEMReward(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(GRPO_DIR))
        cls.plugin = load_module("sapr_grpo_plugin_for_opd_test", GRPO_DIR / "plugin.py")

    def test_exact_match_with_aliases(self):
        reward = self.plugin.SaprEMORM()
        result = reward(
            [
                "So the answer is <answer>The Beatles</answer>",
                "So the answer is <answer>John Lennon</answer>",
                "No final answer",
            ],
            golden_answers=[
                ["Beatles", "The Beatles"],
                ["Paul McCartney"],
                ["anything"],
            ],
        )
        self.assertEqual(result, [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
