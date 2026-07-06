#!/usr/bin/env python3
import argparse
import os
import subprocess


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))

    parser = argparse.ArgumentParser(
        description="Train the Wikitext progressive-growing top-level PC."
    )
    parser.add_argument("--julia-project", default=repo_root)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument(
        "--data-root",
        default=os.path.abspath(os.path.join(here, "..", "progressive_growing", "data")),
        help="Directory containing data_<dataset>/data_trn.npy and data_val.npy",
    )
    parser.add_argument(
        "--temp-root",
        default=os.path.abspath(os.path.join(here, "..", "progressive_growing", "temp")),
        help="Directory containing temp_<dataset>/final_pcs from the low-level PG run",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-independent-clusters", type=int, default=200)
    parser.add_argument("--num-init-clusters", type=int, default=2)
    parser.add_argument("--num-final-clusters", type=int, default=4)
    parser.add_argument("--fname-idx", type=int, default=4)
    parser.add_argument("--num-latents", type=int, default=64)
    parser.add_argument("--num-tr-samples", type=int, default=20_000)
    parser.add_argument("--num-val-samples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs1", type=int, default=10)
    parser.add_argument("--num-epochs2", type=int, default=10)
    args = parser.parse_args()

    env = os.environ.copy()
    julia_gpu = args.gpu
    if "CUDA_VISIBLE_DEVICES" not in env:
        # If we restrict visibility here, Julia should select the first
        # visible device.  Under Slurm, keep the scheduler-provided setting.
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        julia_gpu = 0

    script = os.path.join(os.path.dirname(__file__), "progressive_growing_top.jl")
    cmd = [
        "julia",
        f"--project={args.julia_project}",
        script,
        "--dataset",
        args.dataset,
        "--data-root",
        args.data_root,
        "--temp-root",
        args.temp_root,
        "--gpu",
        str(julia_gpu),
        "--num-independent-clusters",
        str(args.num_independent_clusters),
        "--num-init-clusters",
        str(args.num_init_clusters),
        "--num-final-clusters",
        str(args.num_final_clusters),
        "--fname-idx",
        str(args.fname_idx),
        "--num-latents",
        str(args.num_latents),
        "--num-tr-samples",
        str(args.num_tr_samples),
        "--num-val-samples",
        str(args.num_val_samples),
        "--batch-size",
        str(args.batch_size),
        "--num-epochs1",
        str(args.num_epochs1),
        "--num-epochs2",
        str(args.num_epochs2),
    ]
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
