#!/usr/bin/env python3
"""Print the largest ModernBERT sentence token counts in English Wikipedia."""

from __future__ import annotations

import argparse
import heapq
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "wikimedia/wikipedia"
DEFAULT_CONFIG = "20231101.en"
DEFAULT_SPLIT = "train"
DEFAULT_TOKENIZER = "answerdotai/ModernBERT-base"
DEFAULT_CHECKPOINT = Path("top_wikipedia_sentence_lengths.checkpoint.json")


@dataclass(frozen=True)
class RunIdentity:
    """Settings that determine which sentence lengths a run computes."""

    dataset: str
    config: str
    split: str
    tokenizer: str
    top_n: int


@dataclass(frozen=True)
class Checkpoint:
    """State required to continue an interrupted dataset stream."""

    identity: RunIdentity
    processed_articles: int
    top_lengths: list[int]


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    """Atomically persist checkpoint state as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    payload = {
        "version": 1,
        "identity": asdict(checkpoint.identity),
        "processed_articles": checkpoint.processed_articles,
        "top_lengths": checkpoint.top_lengths,
    }
    with temporary_path.open("w", encoding="utf-8") as checkpoint_file:
        json.dump(payload, checkpoint_file, indent=2)
        checkpoint_file.write("\n")
    os.replace(temporary_path, path)


def load_checkpoint(path: Path) -> Checkpoint:
    """Load checkpoint state from JSON."""
    with path.open(encoding="utf-8") as checkpoint_file:
        payload = json.load(checkpoint_file)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported checkpoint version in {path}")
    return Checkpoint(
        identity=RunIdentity(**payload["identity"]),
        processed_articles=int(payload["processed_articles"]),
        top_lengths=[int(length) for length in payload["top_lengths"]],
    )


def update_top_lengths(heap: list[int], lengths: Iterable[int], limit: int) -> None:
    """Merge lengths into a min-heap containing at most the largest ``limit`` values."""
    for length in lengths:
        if len(heap) < limit:
            heapq.heappush(heap, length)
        elif length > heap[0]:
            heapq.heapreplace(heap, length)


def ensure_punkt_data() -> Callable[[str], list[str]]:
    """Return NLTK's English sentence splitter, downloading its data if needed."""
    import nltk

    try:
        nltk.sent_tokenize("Tokenizer data check.", language="english")
    except LookupError:
        # NLTK 3.9+ uses punkt_tab; older releases use punkt.
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


def article_top_lengths(
    text: str,
    split_sentences: Callable[[str], Sequence[str]],
    tokenizer: Callable[..., Mapping[str, Any]],
    *,
    top_n: int,
    batch_size: int,
) -> list[int]:
    """Compute one article's largest sentence lengths as an isolated heap."""
    article_heap: list[int] = []
    sentences = split_sentences(text)
    for start in range(0, len(sentences), batch_size):
        batch = list(sentences[start : start + batch_size])
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
            verbose=False,
        )
        update_top_lengths(article_heap, encoded["length"], top_n)
    return article_heap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream English Wikipedia and print the 10 largest sentence lengths "
            "measured with the ModernBERT tokenizer."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--tokenization-batch-size", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from --checkpoint instead of starting a fresh run.",
    )
    args = parser.parse_args()
    if args.top_n <= 0:
        parser.error("--top-n must be positive")
    if args.tokenization_batch_size <= 0:
        parser.error("--tokenization-batch-size must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    return args


def dataset_total(dataset: Any, split: str) -> int | None:
    """Read a streaming dataset's declared split size when available."""
    split_info = getattr(getattr(dataset, "info", None), "splits", None)
    if split_info is None or split not in split_info:
        return None
    return split_info[split].num_examples


def main() -> None:
    args = parse_args()
    identity = RunIdentity(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        tokenizer=args.tokenizer,
        top_n=args.top_n,
    )

    if args.resume:
        try:
            checkpoint = load_checkpoint(args.checkpoint)
        except FileNotFoundError as error:
            raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}") from error
        if checkpoint.identity != identity:
            raise SystemExit(
                "Checkpoint settings do not match this run. "
                "Use matching arguments or start without --resume."
            )
        resumed_lengths = checkpoint.top_lengths.copy()
        heapq.heapify(resumed_lengths)
        current = Checkpoint(identity, checkpoint.processed_articles, resumed_lengths)
    else:
        current = Checkpoint(identity, processed_articles=0, top_lengths=[])
        save_checkpoint(args.checkpoint, current)

    from datasets import load_dataset
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    split_sentences = ensure_punkt_data()
    tokenizer = AutoTokenizer.from_pretrained(identity.tokenizer, use_fast=True)
    dataset = load_dataset(
        identity.dataset,
        identity.config,
        split=identity.split,
        streaming=True,
    )
    total_articles = dataset_total(dataset, identity.split)
    if current.processed_articles:
        dataset = dataset.skip(current.processed_articles)

    try:
        with tqdm(
            dataset,
            total=total_articles,
            initial=current.processed_articles,
            unit="article",
            desc="Wikipedia",
        ) as articles:
            for article in articles:
                # Keep work local until the article is complete. If interrupted midway,
                # its partial results cannot leak into a resumable checkpoint.
                article_lengths = article_top_lengths(
                    article["text"],
                    split_sentences,
                    tokenizer,
                    top_n=identity.top_n,
                    batch_size=args.tokenization_batch_size,
                )
                next_lengths = current.top_lengths.copy()
                update_top_lengths(next_lengths, article_lengths, identity.top_n)
                # One assignment commits both the article count and its results. An
                # interruption before this point leaves the resumable state unchanged.
                current = Checkpoint(
                    identity,
                    current.processed_articles + 1,
                    next_lengths,
                )

                if current.processed_articles % args.checkpoint_every == 0:
                    save_checkpoint(args.checkpoint, current)
    except (Exception, KeyboardInterrupt):
        save_checkpoint(args.checkpoint, current)
        raise

    save_checkpoint(args.checkpoint, current)
    print(sorted(current.top_lengths, reverse=True))


if __name__ == "__main__":
    main()
