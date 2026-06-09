import argparse
import os
import sys
import warnings

import torch
from beartype import beartype
from huggingface_loader import TokenizedChunkDataset, get_hf_dataloader
from jaxtyping import Float, Int
from transformers import AutoModel

sys.path.append("../../src/pixelcnn")
sys.path.append("../../src/vqvae2")
sys.path.append("../../src/utils")
sys.path.append("./src")
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../src/vqvae2")

import numpy as np


def get_dataloader(args):
    train_loader, test_loader = get_hf_dataloader(
        tokenizer_name=args.teacher_model,
        chunk_size=args.chunk_size,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        train_split_ratio=args.train_split_ratio,
        max_chunks=args.max_chunks,
    )
    return train_loader, test_loader


@beartype
def get_data_for_vqclusters(
    args,
    model,
    data_loader: torch.utils.data.DataLoader[TokenizedChunkDataset],
    num_samples: int,
    split: str,
):
    chunk_size = args.chunk_size
    embed_size = args.embed_size
    batch_count = len(data_loader)

    data_dir = args.output_dir
    os.makedirs(data_dir, exist_ok=True)
    tokens = np.lib.format.open_memmap(
        os.path.join(data_dir, f"data_{split}.npy"),
        mode="w+",
        dtype=np.int32,
        shape=(num_samples, chunk_size),
    )
    features = np.lib.format.open_memmap(
        os.path.join(data_dir, f"idfeat_{split}.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, chunk_size, embed_size),
    )

    with torch.no_grad():
        model.eval()
        chunk_count = 0
        print(
            f"{split}: extracting {num_samples} chunks in {len(data_loader)} batches",
            flush=True,
        )

        for batch_index, chunk_batch in enumerate(data_loader, start=0):
            remaining = num_samples - chunk_count
            if remaining <= 0:
                break
            if chunk_batch.shape[0] > remaining:
                chunk_batch = chunk_batch[:remaining]

            chunk_batch_cpu: Int[torch.Tensor, "batch_size chunk_size"] = (
                chunk_batch.detach().cpu()
            )
            chunk_batch_gpu: Int[torch.Tensor, "batch_size chunk_size"] = (
                chunk_batch.to(args.device)
            )

            batch_features: Float[torch.Tensor, "batch_size chunk_size embed_size"] = (
                model(input_ids=chunk_batch_gpu).last_hidden_state
            )

            batch_size = chunk_batch_cpu.shape[0]
            next_chunk_count = chunk_count + batch_size
            suffix_sums = torch.flip(
                torch.cumsum(torch.flip(batch_features, dims=[1]), dim=1), dims=[1]
            )
            suffix_lengths = torch.arange(
                chunk_size,
                0,
                -1,
                device=batch_features.device,
                dtype=batch_features.dtype,
            ).view(1, chunk_size, 1)
            suffix_features = suffix_sums / suffix_lengths

            tokens[chunk_count:next_chunk_count] = chunk_batch_cpu.numpy().astype(
                np.int32, copy=False
            )
            features[chunk_count:next_chunk_count] = (
                suffix_features.detach().cpu().numpy().astype(np.float32, copy=False)
            )

            chunk_count = next_chunk_count

            print(
                f"{split}: batch {batch_index + 1}/{len(data_loader)} saved chunks through {chunk_count}",
                flush=True,
            )

        if chunk_count != num_samples:
            raise ValueError(f"Expected {num_samples} chunks, wrote {chunk_count}")

        tokens.flush()
        features.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-clusters", "-c", type=int, default=1024)
    parser.add_argument("--num-skipped-scales", type=int, default=1)
    parser.add_argument("--ll-ratio", "-r", type=float, default=1e-6)
    parser.add_argument("--embed-size", type=int, default=768)
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--teacher-model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Max number of rows to load from the dataset",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Max number of token chunks to process after tokenization",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "progressive_growing",
            "data",
            "data_wikitext",
        ),
        help="Directory to save progressive growing data",
    )
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=0.8,
        help="Fraction of data used for training (rest goes to test)",
    )

    args: argparse.Namespace = parser.parse_args()

    # Task identifier
    args.log_path = f"../../train_logs/wikitext_logs/c{args.n_clusters}_metric-ll"

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    model_id: str = args.teacher_model

    model = AutoModel.from_pretrained(model_id)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")
    args.device = device

    # Get wikitext dataloaders
    train_loader, test_loader = get_dataloader(args)

    print(train_loader.dataset)

    # Get progressive growing dataset
    get_data_for_vqclusters(
        args, model, train_loader, len(train_loader.dataset), split="trn"
    )  # type: ignore
    get_data_for_vqclusters(
        args, model, test_loader, len(test_loader.dataset), split="val"
    )  # type: ignore


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
