# AnyGroundBench

This repository provides the inference and evaluation code for **AnyGroundBench**, a benchmark for domain adaptation in video grounding. 

AnyGroundBench evaluates three grounding tasks:

- `temporal`: predict the temporal segment for a text query.
- `spatial`: predict bounding boxes for a query-specified target.
- `spatio_temporal`: predict both the temporal segment and target boxes.

## Repository Layout

```text
.
|-- README.md
|-- test.py                         # Main inference entry point
|-- data/                           # Dataset metadata, videos, and eval inputs
|-- envs/                           # uv environments for each model family
|-- results/                        # Inference outputs
|-- scripts/
|   |-- inference/
|   |   `-- temporal_grounding.sh    # Example inference commands
|   `-- eval/
|       `-- evaluation.sh            # Example evaluation commands
`-- src/
    |-- datasets/                   # Dataset path helpers
    |-- eval/                       # Metric implementations
    |-- message_builders/           # Model-specific message formatting
    |-- models/                     # Model loading and generation wrappers
    |-- prompts/                    # Task prompts
    |-- timestamp/                  # Timestamp and duration prompt utilities
    |-- utils/                      # Runtime utilities
    `-- video_icl/                  # Temporal, spatial, and spatio-temporal inference
```

## Setup

Install `uv` first, then run commands from the repository root.

Each model family has its own environment under `envs/`:

- `envs/internvl` for InternVL.
- `envs/qwen` for Qwen3-VL.
- `envs/qwen3_5` for Qwen3.5.
- `envs/eagle` for Eagle.
- `envs/gpt` for OpenAI models.
- `envs/gemini` for Gemini models.
- `envs/download` for data utilities and evaluation.

For API-based models, set the required key before inference:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

## Data Tree

Datasets are read from `./data`. Each dataset should contain metadata JSON files and videos. The standard zero-shot layout is:

```text
data/
`-- <dataset_name>/
    |-- meta-data/
    |   |-- t_train.json
    |   |-- t_test.json
    |   |-- s_train.json
    |   |-- s_test.json
    |   |-- st_train.json
    |   `-- st_test.json
    `-- videos/
        |-- 0/
        |   `-- *.mp4
        `-- clips4spatial/
            `-- *.mp4
```

Supported dataset names in the code are:

```text
american_football
animal_kingdom
cholectrack20
dota
egosurgery
enigma
meccano
mouse
multisports
uca
```

The current release includes the `american_football` layout:

```text
data/american_football/
|-- meta-data/
|   |-- s_train.json
|   |-- s_test.json
|   |-- st_train.json
|   |-- st_test.json
|   |-- t_train.json
|   `-- t_test.json
`-- videos/
    |-- 0/
    |   `-- *.mp4
    `-- clips4spatial/
        `-- *.mp4
```

## Running Inference

The main entry point is `test.py`.

Example zero-shot spatio-temporal inference on American football with InternVL:

```bash
uv run --project envs/internvl python test.py \
  --task spatio_temporal \
  --dataset_name american_football \
  --split_num 0 \
  --n_shot 0 \
  --method uniform_sampling \
  --fps 1 \
  --seed 42 \
  --query_max_frames_num 120 \
  --support_max_frames_num 120 \
  --device cuda:0 \
  --model_name internvl \
  --model_id OpenGVLab/InternVL3_5-8B-hf
```

Run the bundled inference examples:

```bash
bash scripts/inference/temporal_grounding.sh
```

Although the script name is `temporal_grounding.sh`, the current commands in that script run `--task spatio_temporal` on `--dataset_name american_football`.

Useful arguments:

- `--task`: one of `temporal`, `spatial`, or `spatio_temporal`.
- `--dataset_name`: dataset key under `data/`.
- `--model_name`: model wrapper name, such as `internvl`, `qwen3`, `qwen3_5`, `eagle`, `gpt`, or `gemini`.
- `--model_id`: model identifier used by the wrapper.
- `--n_shot`: number of demonstrations. Use `0` for zero-shot.
- `--target_query_ids`: run only selected query IDs.
- `--max_test_samples`: limit the number of evaluated test samples.
- `--resize`: resize video inputs for `gpt` or `gemini`.

Inference outputs are written to:

```text
results/temporal/<timestamp>.json
results/spatial/<timestamp>.json
results/spatio_temporal/<timestamp>.json
```

If a sample fails, an error file may also be written next to the prediction JSON.

## Running Evaluation

Evaluation commands are collected in:

```bash
bash scripts/eval/evaluation.sh
```

The evaluation script expects prediction JSON files and runs the task-specific bundle evaluators with `envs/download`.
See `src/eval/` for the metric implementations and JSON-to-CSV conversion logic.
