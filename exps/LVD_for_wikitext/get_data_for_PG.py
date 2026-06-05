import argparse
import torch
import sys
import os
import warnings
from jaxtyping import Float, Int
from beartype import beartype

from transformers import AutoTokenizer, AutoModel

from huggingface_loader import TokenizedChunkDataset, get_hf_dataloader

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
        tokenizer_name = args.teacher_model,
        chunk_size = args.chunk_size,
        dataset_name = args.dataset,
        batch_size = args.batch_size,
        max_rows = args.max_rows,
        train_split_ratio = args.train_split_ratio,
        max_chunks = args.max_chunks
    )
    return train_loader, test_loader



@beartype
def get_data_for_vqclusters(args, model, data_loader: torch.utils.data.DataLoader[TokenizedChunkDataset],num_samples: int,split: str):

    data_dir = args.output_dir
    os.makedirs(data_dir, exist_ok=True)

    num_positions = num_samples * args.chunk_size

    all_chunks = np.lib.format.open_memmap(
        os.path.join(data_dir, f"data_{split}.npy"),
        mode="w+",
        dtype=np.int32,
        shape=(num_positions,),
    )
    yz_feats = np.lib.format.open_memmap(
        os.path.join(data_dir, f"idfeat_{split}.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(num_positions, args.embed_size),
    )
    
    with torch.no_grad():
        model.eval()
        sample_idx = 0
        chunk_count = 0
        print(f"{split}: extracting {num_samples} chunks in {len(data_loader)} batches", flush=True)

        for batch_idx, chunk_batch in enumerate(data_loader, start=1):
            chunk_batch_cpu: Int[torch.Tensor, "batch_size chunk_size"] = chunk_batch.detach().cpu()
            chunk_batch: Int[torch.Tensor, "batch_size chunk_size"] = chunk_batch.to(args.device)

            # DataLoader returns batches; each row is one token chunk.
            for chunk_offset, chunk in enumerate(chunk_batch):
                chunk_cpu: Int[torch.Tensor, "chunk_size"] = chunk_batch_cpu[chunk_offset]

                for i in range(chunk.size(0)):
                    suffix = chunk[i:].unsqueeze(0)
                    quant: Float[torch.Tensor, "embed_size"] = model(input_ids=suffix).last_hidden_state.mean(dim=1).squeeze(0)  # type: ignore
                    feats = quant.detach().cpu()

                    all_chunks[sample_idx] = chunk_cpu[i].item()
                    yz_feats[sample_idx, :] = feats.numpy()
                    sample_idx += 1

                chunk_count += 1

            print(f"{split}: batch {batch_idx}/{len(data_loader)} saved chunks through {chunk_count}", flush=True)

        if sample_idx != num_positions:
            raise ValueError(f"Expected {num_positions} token positions, wrote {sample_idx}")

    all_chunks.flush()
    yz_feats.flush()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type = int, default = 0)
    parser.add_argument("--batch-size", type = int, default = 1024)
    parser.add_argument("--n-clusters","-c", type = int, default = 1024)
    parser.add_argument("--num-skipped-scales", type = int, default = 1)
    parser.add_argument("--ll-ratio", "-r",type = float, default = 1e-6)
    parser.add_argument("--embed-size", type = int, default = 768)
    parser.add_argument("--dataset", default = "wikimedia/wikipedia")
    parser.add_argument("--teacher-model", default = "answerdotai/ModernBERT-base")
    parser.add_argument("--chunk-size", type = int, default = 32)
    parser.add_argument("--max-rows", type = int, default = None, help = "Max number of rows to load from the dataset")
    parser.add_argument("--max-chunks", type = int, default = None, help = "Max number of token chunks to process after tokenization")
    parser.add_argument("--output-dir", default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "progressive_growing", "data", "data_wikitext"), help = "Directory to save progressive growing data")
    parser.add_argument("--train-split-ratio", type = float, default = 0.8, help = "Fraction of data used for training (rest goes to test)")


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

    #Get progressive growing dataset
    get_data_for_vqclusters(args,model,train_loader,len(train_loader.dataset),split='trn') # type: ignore
    get_data_for_vqclusters(args,model,test_loader,len(test_loader.dataset),split='val') # type: ignore
    print("> Data for progressive growing saved")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
