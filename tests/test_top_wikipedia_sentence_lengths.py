from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.top_wikipedia_sentence_lengths import (
    Checkpoint,
    RunIdentity,
    load_checkpoint,
    save_checkpoint,
    update_top_lengths,
)


class UpdateTopLengthsTests(unittest.TestCase):
    def test_keeps_only_the_largest_lengths(self) -> None:
        top_lengths: list[int] = []

        update_top_lengths(top_lengths, [4, 12, 7, 3, 20, 9], limit=3)

        self.assertEqual(sorted(top_lengths, reverse=True), [20, 12, 9])

    def test_preserves_duplicate_length_occurrences(self) -> None:
        top_lengths: list[int] = []

        update_top_lengths(top_lengths, [5, 11, 11, 7], limit=3)

        self.assertEqual(sorted(top_lengths, reverse=True), [11, 11, 7])


class CheckpointTests(unittest.TestCase):
    def test_round_trip_preserves_resume_state(self) -> None:
        identity = RunIdentity(
            dataset="wikimedia/wikipedia",
            config="20231101.en",
            split="train",
            tokenizer="answerdotai/ModernBERT-base",
            top_n=10,
        )
        checkpoint = Checkpoint(
            identity=identity,
            processed_articles=12_345,
            top_lengths=[101, 98, 87],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.json"
            save_checkpoint(path, checkpoint)

            self.assertEqual(load_checkpoint(path), checkpoint)


if __name__ == "__main__":
    unittest.main()
