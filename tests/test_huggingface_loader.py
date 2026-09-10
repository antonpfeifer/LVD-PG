from __future__ import annotations

import unittest
from unittest.mock import patch

from exps.LVD_for_wikitext import huggingface_loader


class CountingDataset:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.seen = 0

    def __iter__(self):
        for text in self.texts:
            self.seen += 1
            yield {"text": text}


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text: str, **_kwargs) -> list[int]:
        return [1] * (40 if text == "too long" else 2)


class SentenceDataLoaderTests(unittest.TestCase):
    def test_stops_reading_dataset_after_collecting_requested_sentences(self) -> None:
        dataset = CountingDataset(["too long", "accepted", "unnecessary"])

        with (
            patch.object(huggingface_loader, "load_dataset", return_value=dataset),
            patch.object(
                huggingface_loader.AutoTokenizer,
                "from_pretrained",
                return_value=FakeTokenizer(),
            ),
            patch.object(
                huggingface_loader,
                "_get_sentence_splitter",
                return_value=lambda text: [text],
            ),
        ):
            train_loader, test_loader = huggingface_loader.get_hf_dataloader_sentence(
                max_chunks=1,
                max_sentence_size=32,
                shuffle_train=False,
            )

        self.assertEqual(dataset.seen, 2)
        self.assertEqual(len(train_loader.dataset) + len(test_loader.dataset), 1)


if __name__ == "__main__":
    unittest.main()
