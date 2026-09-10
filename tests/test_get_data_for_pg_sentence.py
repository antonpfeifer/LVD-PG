from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).parents[1] / "exps" / "LVD_for_wikitext"
sys.path.insert(0, str(EXPERIMENT_DIR))

import get_data_for_PG_sentence  # noqa: E402


class FakeTokenizer:
    pad_token_id = 0


class BFloat16Model:
    tokenizer = FakeTokenizer()

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def eval(self) -> None:
        pass

    def __call__(self, inputs):
        batch_size, sentence_size = inputs["input_ids"].shape
        return {
            "token_embeddings": torch.ones(
                batch_size, sentence_size, 3, dtype=torch.bfloat16
            ),
            "sentence_embedding": torch.ones(
                batch_size, 3, dtype=torch.bfloat16
            ),
        }


class SingleBatchLoader:
    dataset = [None]

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield torch.tensor([[1, 0]], dtype=torch.long)


class FeatureExtractionTests(unittest.TestCase):
    def test_saves_bfloat16_model_embeddings_as_float32(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            get_data_for_PG_sentence.get_data_for_clusters(
                max_sentence_size=2,
                model=BFloat16Model(),
                split="train",
                data_loader=SingleBatchLoader(),
                output_dir=output_dir,
            )

            token_features = np.load(Path(output_dir) / "tokenfeat_train.npy")
            sentence_features = np.load(Path(output_dir) / "sentencefeat_train.npy")

        self.assertEqual(token_features.dtype, np.float32)
        self.assertEqual(sentence_features.dtype, np.float32)
        np.testing.assert_array_equal(token_features, np.ones((1, 2, 3)))
        np.testing.assert_array_equal(sentence_features, np.ones((1, 3)))


if __name__ == "__main__":
    unittest.main()
