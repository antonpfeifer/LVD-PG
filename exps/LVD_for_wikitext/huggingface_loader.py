import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, Dataset as hf_dataset
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


def get_tokenized_ids(
    dataset_name: str = "wikimedia/wikipedia",
    split: str = "train",
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    max_rows: int | None = None,
) -> list[int]:
    if max_rows is not None:
        dataset: hf_dataset = load_dataset(dataset_name, '20231101.en', split=f"{split}[:{max_rows}]")
    else:
        dataset: hf_dataset = load_dataset(dataset_name, '20231101.en', split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    all_ids: list[int] = []
    for example in dataset:
        ids = tokenizer.encode(example["text"], add_special_tokens=False, verbose=False) # type: ignore
        all_ids.extend(ids)

    return all_ids


def get_hf_dataloader(dataset_name="wikimedia/wikipedia", batch_size=8, shuffle_train=True, shuffle_test=False, chunk_size=32, tokenizer_name="answerdotai/ModernBERT-base", max_rows=None, train_split_ratio=0.8, max_chunks=None):
    all_ids = get_tokenized_ids(dataset_name=dataset_name, split="train", tokenizer_name=tokenizer_name, max_rows=max_rows)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

    return train_loader, test_loader
