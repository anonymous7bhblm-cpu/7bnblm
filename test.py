"""Run VideoICL inference from command-line arguments."""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import argparse
import random
import subprocess

import numpy as np
import torch

from src.utils.setup_logger import set_logger
from src.utils.validate_args import validate_args


def parse_args():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()

    # required identity
    parser.add_argument("--task", type=str, required=True,
                        choices=["temporal", "spatio_temporal", "spatial"])
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None)

    # temporal
    parser.add_argument("--non_interleave_text", action="store_true")
    parser.add_argument("--overlay", type=str, default=None)
    parser.add_argument("--add_prompt", nargs="*", default=[])
    parser.add_argument("--text_only_demonstration", action="store_true")
    parser.add_argument("--max_test_samples", type=int, default=None)

    # task/control args
    parser.add_argument("--n_shot", type=int, default=2)
    parser.add_argument("--split_num", type=int, default=1)
    parser.add_argument(
        "--target_query_ids",
        nargs="*",
        default=None,
        help="Run inference only for the specified query IDs. Example: --target_query_ids test_3 test_10",
    )

    # retrieval args
    parser.add_argument("--alpha", type=float, default=None)

    # video sampling args
    parser.add_argument("--support_max_frames_num", type=int, default=16)
    parser.add_argument("--query_max_frames_num", type=int, default=32)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        metavar="PX",
        help="gpt: maximum long edge (px) for frame JPEGs; no resizing when omitted. "
        "gemini: maximum long edge (px) for re-encoding the full video before upload; "
        "requires ffmpeg; uses the original file when omitted.",
    )

    # runtime args
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    # gemini args
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--gemini_mode", type=str, default="sync",
                        choices=["sync", "batch"])
    parser.add_argument("--gemini_batch_size", type=int, default=10)
    parser.add_argument("--batch_poll_interval", type=int, default=30)
    args = parser.parse_args()
    validate_args(args, parser)
    return args

def set_seed(seed: int):
    """Set random seeds for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Run the inference entry point."""
    args = parse_args()

    log_path = os.path.join("logs", f"inference_{args.model_name}.log")
    os.environ["VIDEO_ICL_LOG_PATH"] = log_path
    logger = set_logger(log_path)

    set_seed(args.seed)
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    logger.info(f"gArgs: {args}. git hash: {git_hash}")


    if args.task == "temporal":
        from src.video_icl.temporal import TemporalVideoICL
        icl = TemporalVideoICL(args)
    elif args.task == "spatio_temporal":
        from src.video_icl.spatio_temporal import SpatioTemporalVideoICL
        icl = SpatioTemporalVideoICL(args)
    elif args.task == "spatial":
        from src.video_icl.spatial import SpatialVideoICL
        icl = SpatialVideoICL(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    try:
        icl.inference()
    except TimeoutError as e:
        if args.model_name == "gemini":
            raise RuntimeError(
                "Gemini request timed out and reached max_retries="
                f"{args.max_retries}. Aborting."
            ) from e
        raise


if __name__ == "__main__":
    main()
