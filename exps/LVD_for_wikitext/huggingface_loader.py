import torch
from datasets import Dataset as hf_dataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TokenizedChunkDataset(Dataset):
    def __init__(self, token_ids: list[int], chunk_size: int = 32):
        self.chunk_size = chunk_size
        n_chunks = len(token_ids) // chunk_size
        trimmed = token_ids[: n_chunks * chunk_size]
        self.chunks = torch.tensor(trimmed, dtype=torch.long).reshape(-1, chunk_size)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


class TokenizedSentenceDataset(Dataset):
    def __init__(
        self,
        tokenized_sentences: list[list[int]],
        max_sentence_size: int,
        pad_token_id: int,
    ):
        self.max_sentence_size = max_sentence_size
        self.sentences = torch.full(
            (len(tokenized_sentences), max_sentence_size),
            pad_token_id,
            dtype=torch.long,
        )
        for index, token_ids in enumerate(tokenized_sentences):
            self.sentences[index, : len(token_ids)] = torch.tensor(
                token_ids, dtype=torch.long
            )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def get_tokenized_ids(
    dataset_name: str = "wikimedia/wikipedia",
    split: str = "train",
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    max_rows: int | None = None,
) -> list[int]:
    dataset: hf_dataset = None

    if max_rows is not None:
        dataset = load_dataset(
            dataset_name, "20231101.en", split=f"{split}[:{max_rows}]"
        )
    else:
        dataset = load_dataset(dataset_name, "20231101.en", split=split)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    all_ids: list[int] = []
    for example in dataset:
        ids = tokenizer.encode(example["text"], add_special_tokens=False, verbose=False)  # type: ignore
        all_ids.extend(ids)

    return all_ids


def get_hf_dataloader(
    dataset_name="wikimedia/wikipedia",
    batch_size=8,
    shuffle_train=True,
    shuffle_test=False,
    chunk_size=32,
    tokenizer_name="answerdotai/ModernBERT-base",
    max_rows=None,
    train_split_ratio=0.8,
    max_chunks=None,
):
    all_ids = get_tokenized_ids(
        dataset_name=dataset_name,
        split="train",
        tokenizer_name=tokenizer_name,
        max_rows=max_rows,
    )

    n_chunks = len(all_ids) // chunk_size
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)

    split_chunks = int(n_chunks * train_split_ratio)
    split_idx = split_chunks * chunk_size
    end_idx = n_chunks * chunk_size
    train_ids = all_ids[:split_idx]
    test_ids = all_ids[split_idx:end_idx]

    train_dataset = TokenizedChunkDataset(train_ids, chunk_size=chunk_size)
    test_dataset = TokenizedChunkDataset(test_ids, chunk_size=chunk_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

    return train_loader, test_loader


def _get_sentence_splitter():
    import nltk

    try:
        nltk.sent_tokenize("Tokenizer data check.", language="english")
    except LookupError:
        for package in ("punkt_tab", "punkt"):
            try:
                nltk.download(package, quiet=True, raise_on_error=True)
                nltk.sent_tokenize("Tokenizer data check.", language="english")
                break
            except (LookupError, ValueError):
                continue
        else:
            raise RuntimeError("Unable to install NLTK Punkt tokenizer data")

    return lambda text: nltk.sent_tokenize(text, language="english")


def get_hf_dataloader_sentence(
    dataset_name="wikimedia/wikipedia",
    batch_size=8,
    shuffle_train=True,
    shuffle_test=False,
    max_sentence_size=32,
    tokenizer_name="answerdotai/ModernBERT-base",
    max_rows=None,
    train_split_ratio=0.8,
    max_chunks=None,
):
    if max_sentence_size <= 0:
        raise ValueError("max_sentence_size must be greater than zero")

    if max_rows is not None:
        dataset = load_dataset(
            dataset_name, "20231101.en", split=f"train[:{max_rows}]"
        )
    else:
        dataset = load_dataset(dataset_name, "20231101.en", split="train")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name!r} does not define a pad token")

    split_sentences = _get_sentence_splitter()
    tokenized_sentences: list[list[int]] = []
    if max_chunks != 0:
        for example in dataset:
            for sentence in split_sentences(example["text"]):
                token_ids = tokenizer.encode(
                    sentence, add_special_tokens=False, verbose=False
                )
                if len(token_ids) <= max_sentence_size:
                    tokenized_sentences.append(token_ids)
                    if (
                        max_chunks is not None
                        and len(tokenized_sentences) >= max_chunks
                    ):
                        break

            if max_chunks is not None and len(tokenized_sentences) >= max_chunks:
                break

    split_idx = int(len(tokenized_sentences) * train_split_ratio)
    train_sentences = tokenized_sentences[:split_idx]
    test_sentences = tokenized_sentences[split_idx:]

    train_dataset = TokenizedSentenceDataset(
        train_sentences,
        max_sentence_size=max_sentence_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    test_dataset = TokenizedSentenceDataset(
        test_sentences,
        max_sentence_size=max_sentence_size,
        pad_token_id=tokenizer.pad_token_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train and len(train_dataset) > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test and len(test_dataset) > 0,
    )

    return train_loader, test_loader
