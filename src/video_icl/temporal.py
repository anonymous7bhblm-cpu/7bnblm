"""Classification Video ICL"""
import json
import os
import re

from tqdm import tqdm

from src.datasets import get_domain_from_dataset_name, get_temporal_info
from src.message_builders import (
    build_demonstration_message_builder,
    build_query_message_builder,
)
from src.prompts import build_prompt
from src.prompts.registry import normalize_model_family
from src.timestamp.not_interleave_text import add_duration, add_timestamp
from src.utils.setup_logger import set_logger
from src.video_icl.base import BaseVideoICL

logger = set_logger()
BRACKET_PAIR_RE = re.compile(r"^\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]$")

class TemporalVideoICL(BaseVideoICL):
    """Grounding Video ICL"""

    @staticmethod
    def _to_normalized_span_text(start_sec: float, end_sec: float, duration_sec: float) -> str:
        """Internal helper for to normalized span text."""
        if duration_sec <= 0:
            return "{<t0.000>,<t0.000>}"
        start = min(max(start_sec / duration_sec, 0.0), 1.0)
        end = min(max(end_sec / duration_sec, 0.0), 1.0)
        if end < start:
            start, end = end, start
        return f"{{<t{start:.3f}>,<t{end:.3f}>}}"

    def _run_temporal_inference_with_retry(self, messages):
        """Internal helper for run temporal inference with retry."""
        # proprietary
        if self.model_name == "gemini":
            prediction, prob = self.process_inference(messages)
            return prediction, prob, None
        if self.model_name == "gpt":
            prediction, prob = self.process_inference(messages)
            return prediction, prob, None

        # OSS
        if self.model_name in {"qwen3", "qwen3_5", "internvl", "eagle"}:
            raw_outputs = []
            last_prob = None
            for _ in range(5): # Run up to 5 times.
                prediction, prob = self.process_inference(messages)
                last_prob = prob
                if isinstance(prediction, str) and BRACKET_PAIR_RE.fullmatch(prediction.strip()) is not None:
                    return prediction, prob, None
                raw_outputs.append(prediction)

            return raw_outputs[-1], last_prob, raw_outputs

        # specialist
        if self.model_name == "llava_st":
            prediction, prob = self.process_inference(messages)
            return prediction, prob, None

        raise ValueError(f"Unsupported temporal retry model: {self.model_name}")

    def __init__(self, args):
        """Initialize the instance."""
        super().__init__(args)
        self.domain = get_domain_from_dataset_name(self.dataset_name)

        dataset_info = get_temporal_info(self.dataset_name)
        self.annotation_dir = dataset_info["annotation_dir"]

        if self.n_shot == 0:
            self.meta_data_dir = dataset_info["meta_data_dir"]
            self.video_data_dir = dataset_info["video_data_dir"]
        else:
            self.meta_data_dir = os.path.join("data", self.domain, "meta-data")
            self.video_data_dir = os.path.join("data", self.domain, "videos")
        self.train_video_to_entries = {}
        with open(os.path.join(self.meta_data_dir, "t_train.json"), encoding="utf-8") as f:
            train_data = json.load(f)
        for train_id, entry in train_data.items():
            start_str, end_str = entry["temporal_range"].split()
            self.train_video_to_entries[train_id] = {
                "video_name": f'{entry["video_name"]}.mp4',
                "gt_time_range": [float(start_str), float(end_str)],
                "text": entry["text"].strip(),
            }

        self.test_video_to_entries = {}
        with open(os.path.join(self.meta_data_dir, "t_test.json"), encoding="utf-8") as f:
            test_data = json.load(f)
        for test_id, entry in test_data.items():
            start_str, end_str = entry["temporal_range"].split()
            self.test_video_to_entries[test_id] = {
                "video_name": f'{entry["video_name"]}.mp4',
                "gt_time_range": [float(start_str), float(end_str)],
                "text": entry["text"].strip(),
            }


        self.compress = args.compress
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.non_interleave_text = args.non_interleave_text
        self.overlay = args.overlay
        self.add_prompt = args.add_prompt
        self.text_only_demonstration = args.text_only_demonstration
        self.text_only = False
        if self.text_only_demonstration:
            self.text_only = True
        self.max_test_samples = args.max_test_samples
        self.target_query_ids = set(args.target_query_ids or [])

        self.build_demonstration_message = build_demonstration_message_builder(
            self.model_name,
            text_only=self.text_only_demonstration,
        )
        self.build_query_message = build_query_message_builder(
            self.model_name,
            text_only=self.text_only
        )

    def prepare_messages(self, demonstrations, query_id):
        """Prepare support messages as demonstrations."""
        demonstration_messages = []
        for demonstration in demonstrations:
            demonstration_info = self.train_video_to_entries[demonstration]
            video_name = demonstration_info["video_name"]
            demonstration_video_path = os.path.join(self.video_data_dir, str(self.split_num), video_name)
            ground_truth = self.train_video_to_entries[demonstration]["gt_time_range"]
            if normalize_model_family(self.model_name) == "llava_st":
                from src.timestamp.get_timestamp import get_duration  # Lazy import.
                duration_sec = get_duration(demonstration_video_path)
                ground_truth = self._to_normalized_span_text(
                    start_sec=float(ground_truth[0]),
                    end_sec=float(ground_truth[1]),
                    duration_sec=duration_sec,
                )
            demonstration_temporal_prompt = build_prompt(
                task=self.task,
                model_name=self.model_name,
                query_text=demonstration_info["text"],
            ).text
            if "timestamp" in self.add_prompt:
                timestamp_prompt = add_timestamp(
                    video_path = demonstration_video_path,
                    fps = self.fps,
                    max_frames = self.support_max_frames_num,
                    method = self.method,
                    model_name = self.model_name
                )
                demonstration_temporal_prompt = timestamp_prompt + demonstration_temporal_prompt
            if "duration" in self.add_prompt:
                duration_prompt = add_duration(video_path=demonstration_video_path)
                demonstration_temporal_prompt = duration_prompt + demonstration_temporal_prompt
            demonstration_messages += self.build_demonstration_message(
                support_video_path=demonstration_video_path,
                support_max_frames_num=self.support_max_frames_num,
                fps=self.fps,
                question_sample=demonstration_temporal_prompt,
                ground_truth=ground_truth,
            )

        query_info = self.test_video_to_entries[query_id]
        query_video_path = os.path.join(self.video_data_dir, str(self.split_num), query_info["video_name"])
        query_temporal_prompt = build_prompt(
            task=self.task,
            model_name=self.model_name,
            query_text=query_info["text"],
        ).text
        if "timestamp" in self.add_prompt:
            timestamp_prompt = add_timestamp(
                video_path = query_video_path,
                fps = self.fps,
                max_frames = self.query_max_frames_num,
                method = self.method,
                model_name = self.model_name
            )
            query_temporal_prompt = timestamp_prompt + query_temporal_prompt
        if "duration" in self.add_prompt:
            duration_prompt = add_duration(
                video_path=query_video_path,
            )
            query_temporal_prompt = duration_prompt + query_temporal_prompt
        query_messages = self.build_query_message(
            query_video_path=query_video_path,
            query_max_frames_num=self.query_max_frames_num,
            fps=self.fps,
            question_sample=query_temporal_prompt,
        )
        messages = demonstration_messages + query_messages
        # print(messages)
        return messages

    def get_data(self):
        """Get data."""
        if self.n_shot == 0:
            test_videos = list(self.test_video_to_entries.keys())
            if self.target_query_ids:
                test_videos = [sample_id for sample_id in test_videos if sample_id in self.target_query_ids]
            if self.max_test_samples is not None:
                test_videos = test_videos[:self.max_test_samples]
            return {}, test_videos

        simrank_path = os.path.join("data", self.domain, "simrank", f"video_top100_alpha{self.alpha}.json")
        with open(simrank_path) as f:
            annotation = json.load(f)

        similarity_rank = {}
        for i in annotation:
            similarity_rank[i['test_sample']] = i['train_examples']

        test_videos = [row["test_sample"] for row in annotation]
        if self.target_query_ids:
            test_videos = [sample_id for sample_id in test_videos if sample_id in self.target_query_ids]
        if self.max_test_samples is not None:
            test_videos = test_videos[:self.max_test_samples]
        return similarity_rank, test_videos

    def inference(self):
        """Run the inference pipeline."""
        sim_rank, test_videos = self.get_data()

        valid_count = 0
        prediction_records = []
        errors = []
        for query_id in tqdm(test_videos):
            try:
                if self.n_shot == 0:
                    demonstrations = []
                else:
                    candidates = sim_rank[query_id.split('/')[-1]]
                    demonstrations = candidates[:self.n_shot]
                messages = self.prepare_messages(demonstrations, query_id)
                prediction, _, retry_raw_outputs = self._run_temporal_inference_with_retry(messages)
                ground_truth_start = self.test_video_to_entries[query_id]["gt_time_range"][0]
                ground_truth_end = self.test_video_to_entries[query_id]["gt_time_range"][1]
                ground_truth = f"[{ground_truth_start}, {ground_truth_end}]"
                # print(f"gt: {ground_truth}, pred: {prediction}")
                record = {
                    "query_id": query_id,
                    "demonstrations": demonstrations,
                    "gt": ground_truth,
                    "raw_pred": prediction,
                    "method": self.method,
                    "model_id": self.model_id,
                    "dataset_name": self.dataset_name,
                    "n_shot": self.n_shot,
                    "split_num": self.split_num,
                    "text_only_demonstration": self.text_only_demonstration,
                    "query_video_path": self.test_video_to_entries[query_id]["video_name"],
                }
                if retry_raw_outputs is not None:
                    record["raw_pred_retry_outputs"] = retry_raw_outputs
                prediction_records.append(record)

                valid_count += 1

            except Exception as e:
                query_entry = self.test_video_to_entries.get(query_id, {})
                short_error = f"{type(e).__name__}: {str(e)}"
                errors.append({
                    "query_id": query_id,
                    "video_name": query_entry.get("video_name"),
                    "error": short_error,
                })

        avg_context_num = round(self.all_context_num / self.context_count, 5) if self.context_count > 0 else 0.0
        logger.info(f"Avg Context Number: {avg_context_num}")
        logger.info(f"Valid Count: {valid_count}")

        os.makedirs("results/temporal", exist_ok=True)
        result_path = os.path.join("results", "temporal", f"{self.now}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(prediction_records, f, indent=2, ensure_ascii=False)
        error_path = os.path.join("results", "temporal", f"{self.now}_errors.json")
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        self.upload_run_record(result_path, valid_count=valid_count)
        return result_path
