import argparse
import os

import torch

import numpy as np

from huggingface_loader import get_hf_dataloader_sentence
from jaxtyping import Float, Int
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_data_for_clusters(
    max_sentence_size: int,
    model: SentenceTransformer,
    split: str,
    data_loader,
    output_dir: str,
):
    num_sentences = len(data_loader.dataset)
    embed_size = model.get_embedding_dimension()
    if embed_size is None:
        raise ValueError("The sentence transformer does not expose its embedding size")

    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("The sentence transformer tokenizer does not define a pad token")

    import json

    with open(
        os.path.join(output_dir, "model_metadata.json"),
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            {
                "model_name": model.model_name_or_path,
                "pad_token_id": pad_token_id,
                "vocab_size": model.tokenizer.vocab_size,
            },
            metadata_file,
        )

    tokens = np.lib.format.open_memmap(
        os.path.join(output_dir, f"data_{split}.npy"),
        mode="w+",
        dtype=np.int32,
        shape=(num_sentences, max_sentence_size),
    )
    token_features = np.lib.format.open_memmap(
        os.path.join(output_dir, f"tokenfeat_{split}.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(num_sentences, max_sentence_size, embed_size),
    )
    sentence_features = np.lib.format.open_memmap(
        os.path.join(output_dir, f"sentencefeat_{split}.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(num_sentences, embed_size),
    )

    with torch.no_grad():
        model.eval()
        sentence_count = 0
        print(
            f"{split}: extracting {num_sentences} sentences in "
            f"{len(data_loader)} batches",
            flush=True,
        )
        for batch_index, sentence_batch in enumerate(data_loader, start=0):
            remaining = num_sentences - sentence_count
            if remaining <= 0:
                break
            if sentence_batch.shape[0] > remaining:
                sentence_batch = sentence_batch[:remaining]

            sentence_batch_cpu: Int[torch.Tensor, "batch_size max_sentence_size"] = (
                sentence_batch.detach().cpu()
            )
            sentence_batch_gpu: Int[torch.Tensor, "batch_size max_sentence_size"] = (
                sentence_batch.to(device)
            )
            attention_mask = sentence_batch_gpu.ne(pad_token_id).long()
            model_output = model(
                {
                    "input_ids": sentence_batch_gpu,
                    "attention_mask": attention_mask,
                }
            )
            batch_token_features: Float[
                torch.Tensor, "batch_size max_sentence_size embed_size"
            ] = model_output["token_embeddings"]
            batch_sentence_features: Float[
                torch.Tensor, "batch_size embed_size"
            ] = model_output["sentence_embedding"]

            current_batch_size = sentence_batch_cpu.shape[0]
            next_sentence_count = sentence_count + current_batch_size
            tokens[sentence_count:next_sentence_count] = (
                sentence_batch_cpu.numpy().astype(np.int32, copy=False)
            )
            token_features[sentence_count:next_sentence_count] = (
                batch_token_features.detach()
                .cpu()
                .to(dtype=torch.float32)
                .numpy()
            )
            sentence_features[sentence_count:next_sentence_count] = (
                batch_sentence_features.detach()
                .cpu()
                .to(dtype=torch.float32)
                .numpy()
            )
            sentence_count = next_sentence_count

            print(
                f"{split}: batch {batch_index + 1}/{len(data_loader)} "
                f"saved sentences through {sentence_count}",
                flush=True,
            )

        if sentence_count != num_sentences:
            raise ValueError(
                f"Expected {num_sentences} sentences, wrote {sentence_count}"
            )

        tokens.flush()
        token_features.flush()
        sentence_features.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sentence_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="google/embeddinggemma-300m",
    )
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--train-split-ratio", type=float, default=0.8)
    args = parser.parse_args()

    # make output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # load model
    model = SentenceTransformer(args.teacher_model, device=device)

    # load data
    train_loader, test_loader = get_hf_dataloader_sentence(
        tokenizer_name=args.teacher_model,
        max_sentence_size=args.max_sentence_size,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train_split_ratio=args.train_split_ratio,
        max_chunks=args.max_sentences,
    )

    # extract features for train and test
    get_data_for_clusters(
        max_sentence_size=args.max_sentence_size,
        model=model,
        split="train",
        data_loader=train_loader,
        output_dir=args.output_dir,
    )

    get_data_for_clusters(
        max_sentence_size=args.max_sentence_size,
        model=model,
        split="test",
        data_loader=test_loader,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
