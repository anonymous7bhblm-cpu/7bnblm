"""Argument validation utilities."""
import argparse

CLASSIFICATION_DATASETS = {"crime", "egopet", "drive", "xsports", "mammalps", "gesture"}
CAPTIONING_DATASETS = {"capera", "bora"}
OPEN_QA_DATASETS = {"sports_qa", "intentqa", "nextqa"}
TEMPORAL_DATASETS = {
    "animal_kingdom",
    "american_football",
    "charades",
    "cholectrack20",
    "dota",
    "egosurgery",
    "enigma",
    "meccano",
    "mouse",
    "multisports",
    "timelens_charades",
    "uca",
}
SPATIO_TEMPORAL_DATASETS = {
    "animal_kingdom",
    "american_football",
    "cholectrack20",
    "dota",
    "egosurgery",
    "enigma",
    "meccano",
    "mouse",
    "multisports",
    "uca",
}
ACTION_SEGMENTATION_DATASETS = {"breakfast"}
NON_REINFORCE_METHODS = {"uniform_sampling", "random", "random_highlight", "best_loss", "worst_loss", "visual_overlay"}
ALLOWED_COMPRESS = {None, "sim_token_score", "first_sim_then_token_score"}
ALLOWED_ADD_PROMPTS = {"timestamp", "duration"}
TIMESTAMP_SUPPORTED_MODELS = {"qwen3", "qwen3_5", "internvl", "eagle", "llava_st"}
TIMESTAMP_SUPPORTED_METHODS = {"uniform_sampling", "visual_overlay"}
TEXT_ONLY_DEMONSTRATION_SUPPORTED_MODELS = {"qwen2", "qwen2_5", "qwen3", "qwen3_5", "internvl", "eagle", "llava_st", "gemini"}
TEXT_ONLY_DEMONSTRATION_SUPPORTED_METHODS = {"uniform_sampling"}

def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate parsed arguments."""
    errors = []

    # task <-> dataset consistency
    if args.task == "classification" and args.dataset_name not in CLASSIFICATION_DATASETS:
        errors.append(f"task=classification requires dataset_name in {sorted(CLASSIFICATION_DATASETS)}")
    if args.task == "captioning" and args.dataset_name not in CAPTIONING_DATASETS:
        errors.append(f"task=captioning requires dataset_name in {sorted(CAPTIONING_DATASETS)}")
    if args.task == "open_qa" and args.dataset_name not in OPEN_QA_DATASETS:
        errors.append(f"task=open_qa requires dataset_name in {sorted(OPEN_QA_DATASETS)}")
    if args.task == "temporal" and args.dataset_name not in TEMPORAL_DATASETS:
        errors.append(f"task=temporal requires dataset_name in {sorted(TEMPORAL_DATASETS)}")
    if args.task in {"spatio_temporal", "spatial"} and args.dataset_name not in SPATIO_TEMPORAL_DATASETS:
        errors.append(f"task={args.task} requires dataset_name in {sorted(SPATIO_TEMPORAL_DATASETS)}")
    if args.task == "action_segmentation" and args.dataset_name not in ACTION_SEGMENTATION_DATASETS:
        errors.append(f"task=action_segmentation requires dataset_name in {sorted(ACTION_SEGMENTATION_DATASETS)}")

    # method-dependent constraints
    if args.threshold is not None and args.top_k is not None:
        errors.append("--threshold and --top_k cannot be used together")

    if args.method in {"random", "random_highlight"}:
        if args.max_random_frames is None:
            errors.append(f"method={args.method} requires --max_random_frames")
        if args.threshold is not None or args.top_k is not None:
            errors.append(f"method={args.method} cannot be used with --threshold/--top_k")
    elif args.method in {"uniform_sampling", "best_loss", "worst_loss", "visual_overlay"}:
        if args.threshold is not None or args.top_k is not None:
            errors.append(f"method={args.method} cannot be used with --threshold/--top_k")
        if args.max_random_frames is not None:
            errors.append(f"method={args.method} cannot be used with --max_random_frames")
    elif args.method == "highlight":
        if args.max_random_frames is not None:
            errors.append("method=highlight cannot be used with --max_random_frames")
    else:
        # reinforce-style methods: exactly one of threshold or top_k is required
        if (args.threshold is None) == (args.top_k is None):
            errors.append("reinforce-style methods require exactly one of --threshold or --top_k")
        if args.max_random_frames is not None:
            errors.append("reinforce-style methods cannot be used with --max_random_frames")
    if args.method == "visual_overlay":
        if args.overlay is None:
            errors.append("method=visual_overlay requires --overlay")
        elif args.overlay not in {"time", "frame"}:
            errors.append("--overlay must be one of ['time', 'frame']")
    elif args.overlay is not None:
        errors.append("--overlay is only supported for method=visual_overlay")

    # task-specific args
    if args.task == "open_qa":
        if args.alpha is None:
            errors.append("task=open_qa requires --alpha")
    elif args.task in {"temporal", "spatio_temporal", "spatial", "action_segmentation"}:
        if args.n_shot == 0 and args.alpha is not None:
            errors.append(f"task={args.task} with n_shot=0 does not support --alpha")
        if args.alpha is None and args.n_shot != 0:
            errors.append(f"task={args.task} requires --alpha")
    elif args.alpha is not None:
        errors.append("--alpha is only supported for task=open_qa, task=temporal, task=spatio_temporal, task=spatial, or task=action_segmentation")

    # compress constraints
    if args.compress not in ALLOWED_COMPRESS:
        errors.append(f"--compress must be one of {sorted(x for x in ALLOWED_COMPRESS if x is not None)}")
    if args.compress == "sim_token_score" and args.gamma is None:
        errors.append("--gamma is required when --compress=sim_token_score")

    # temporal prompt constraints
    add_prompt = getattr(args, "add_prompt", []) or []
    if isinstance(add_prompt, str):
        add_prompt = [add_prompt]

    invalid_add_prompt = sorted(set(add_prompt) - ALLOWED_ADD_PROMPTS)
    if invalid_add_prompt:
        errors.append(f"--add_prompt must contain only {sorted(ALLOWED_ADD_PROMPTS)}")

    if len(add_prompt) != len(set(add_prompt)):
        errors.append("--add_prompt must not contain duplicate values")

    if add_prompt and args.task not in {"temporal", "spatio_temporal", "spatial", "action_segmentation"}:
        errors.append("--add_prompt is only supported for task=temporal, task=spatio_temporal, task=spatial, or task=action_segmentation")

    if "timestamp" in add_prompt:
        if args.model_name not in TIMESTAMP_SUPPORTED_MODELS:
            errors.append(
                f"--add_prompt timestamp requires model_name in {sorted(TIMESTAMP_SUPPORTED_MODELS)}"
            )
        if args.method not in TIMESTAMP_SUPPORTED_METHODS:
            errors.append(
                f"--add_prompt timestamp requires method in {sorted(TIMESTAMP_SUPPORTED_METHODS)}"
            )

    if getattr(args, "text_only_demonstration", False):
        if args.task not in {"temporal", "spatio_temporal", "spatial", "action_segmentation"}:
            errors.append("--text_only_demonstration is only supported for task=temporal, task=spatio_temporal, task=spatial, or task=action_segmentation")
        if args.model_name not in TEXT_ONLY_DEMONSTRATION_SUPPORTED_MODELS:
            errors.append(
                "--text_only_demonstration currently supports only model_name in "
                f"{sorted(TEXT_ONLY_DEMONSTRATION_SUPPORTED_MODELS)}"
            )
        if args.method not in TEXT_ONLY_DEMONSTRATION_SUPPORTED_METHODS:
            errors.append(
                "--text_only_demonstration currently supports only method in "
                f"{sorted(TEXT_ONLY_DEMONSTRATION_SUPPORTED_METHODS)}"
            )

    if errors:
        parser.error("Invalid arguments:\n- " + "\n- ".join(errors))
