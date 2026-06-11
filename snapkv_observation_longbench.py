import json
import os

from attn_heatmap import (
    build_run_dir,
    build_token_entries,
    extract_result_summary,
    sanitize_slug,
)
from models.compression.experiments.snapkv_observation import (
    SnapKVObservationConfig,
    compute_snapkv_observation,
    compute_snapkv_topk_overlap_observation,
    save_snapkv_observation,
    save_snapkv_topk_overlap_observation,
)


class SnapKVObservationRunWriter:
    def __init__(self, root_dir, model_name, out_file, config):
        self.root_dir = root_dir
        self.model_name = model_name
        self.out_file = os.path.abspath(out_file)
        self.config = config
        self.run_dir = build_run_dir(root_dir, out_file)
        self.samples_dir = os.path.join(self.run_dir, "samples")
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.samples = []
        self._next_sample_index = 0
        os.makedirs(self.samples_dir, exist_ok=True)
        self._write_manifest()

    def new_sample(self, item):
        sample_index = self._next_sample_index
        self._next_sample_index += 1
        return SnapKVObservationSampleWriter(self, item, sample_index)

    def register_sample(self, sample_writer):
        sample_summary = {
            "sample_id": sample_writer.sample_id,
            "sample_index": sample_writer.sample_index,
            "sample_path": sample_writer.sample_rel_path,
            "_id": sample_writer.item.get("_id"),
            "domain": sample_writer.item.get("domain"),
            "question": sample_writer.item.get("question"),
            "pred": sample_writer.result_summary.get("pred"),
            "judge": sample_writer.result_summary.get("judge"),
            "prefill_count": len(sample_writer.prefills),
            "prefill_statuses": [prefill["status"] for prefill in sample_writer.prefills],
        }
        self.samples.append(sample_summary)
        self._write_manifest()

    def _write_manifest(self):
        manifest = {
            "run_dir": self.run_dir,
            "model_name": self.model_name,
            "result_path": self.out_file,
            "snapkv_observation_config": {
                "budget": self.config.budget,
                "window_size": self.config.window_size,
                "kernel_size": self.config.kernel_size,
                "max_prefill_tokens": self.config.max_prefill_tokens,
                "hidden_mix_profile_path": self.config.hidden_mix_profile_path,
            },
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as fout:
            json.dump(manifest, fout, ensure_ascii=False, indent=2)


class SnapKVObservationSampleWriter:
    def __init__(self, run_writer, item, sample_index):
        self.run_writer = run_writer
        self.item = dict(item)
        self.sample_index = int(sample_index)
        base_name = sanitize_slug(
            self.item.get("_id", f"sample_{sample_index:04d}"),
            fallback=f"sample_{sample_index:04d}",
        )
        self.sample_id = f"sample_{sample_index:04d}_{base_name}"
        self.sample_dir = os.path.join(self.run_writer.samples_dir, self.sample_id)
        self.sample_rel_path = os.path.relpath(self.sample_dir, self.run_writer.run_dir)
        self.sample_json_path = os.path.join(self.sample_dir, "sample.json")
        self.prefills = []
        self.result_summary = {}
        os.makedirs(self.sample_dir, exist_ok=True)
        self._write_sample_json()

    def capture_prefill(self, model, tokenizer, prompt_text, inputs, label):
        prefill_index = len(self.prefills)
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        record = {
            "prefill_index": prefill_index,
            "label": label,
            "status": "pending",
            "prompt_text": prompt_text,
            "token_count": len(input_ids),
            "token_ids": input_ids,
            "tokens": build_token_entries(tokenizer, input_ids),
            "summary_file": None,
            "npz_file": None,
            "image_files": {},
            "valid_layer_indices": [],
        }
        self.prefills.append(record)
        self._write_sample_json()

        try:
            result = compute_snapkv_observation(
                model=model,
                inputs=inputs,
                config=self.run_writer.config,
            )
            artifact_prefix = f"prefill_{prefill_index:03d}_snapkv_observation"
            paths = save_snapkv_observation(
                result=result,
                output_dir=self.sample_dir,
                prefix=artifact_prefix,
            )
            record["status"] = result.summary.get("status", "saved")
            record["summary_file"] = os.path.basename(paths["summary"])
            record["npz_file"] = os.path.basename(paths["npz"])
            record["image_files"] = {
                name: os.path.basename(path)
                for name, path in paths.get("images", {}).items()
            }
            record["valid_layer_indices"] = result.summary.get("valid_layer_indices", [])
            record["valid_layer_count"] = len(record["valid_layer_indices"])
            if result.summary.get("reason"):
                record["reason"] = result.summary["reason"]
            if result.summary.get("error"):
                record["error"] = result.summary["error"]
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)

        self._write_sample_json()
        return record

    def finalize(self, item):
        self.result_summary = extract_result_summary(item)
        self._write_sample_json()
        self.run_writer.register_sample(self)

    def build_capture_status(self):
        keys = [
            "prefill_index",
            "label",
            "status",
            "token_count",
            "valid_layer_count",
            "summary_file",
            "npz_file",
            "image_files",
            "reason",
            "error",
        ]
        return [{key: record[key] for key in keys if key in record} for record in self.prefills]

    def _write_sample_json(self):
        sample_payload = {
            "sample_id": self.sample_id,
            "sample_index": self.sample_index,
            "item": extract_result_summary(self.item),
            "result": self.result_summary,
            "prefills": self.prefills,
        }
        with open(self.sample_json_path, "w", encoding="utf-8") as fout:
            json.dump(sample_payload, fout, ensure_ascii=False, indent=2)


class SnapKVTopKOverlapRunWriter:
    def __init__(self, root_dir, model_name, out_file, config):
        self.root_dir = root_dir
        self.model_name = model_name
        self.out_file = os.path.abspath(out_file)
        self.config = config
        self.run_dir = f"{build_run_dir(root_dir, out_file)}_topk_overlap"
        self.samples_dir = os.path.join(self.run_dir, "samples")
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.samples = []
        self._next_sample_index = 0
        os.makedirs(self.samples_dir, exist_ok=True)
        self._write_manifest()

    def new_sample(self, item):
        sample_index = self._next_sample_index
        self._next_sample_index += 1
        return SnapKVTopKOverlapSampleWriter(self, item, sample_index)

    def register_sample(self, sample_writer):
        sample_summary = {
            "sample_id": sample_writer.sample_id,
            "sample_index": sample_writer.sample_index,
            "sample_path": sample_writer.sample_rel_path,
            "_id": sample_writer.item.get("_id"),
            "domain": sample_writer.item.get("domain"),
            "question": sample_writer.item.get("question"),
            "pred": sample_writer.result_summary.get("pred"),
            "judge": sample_writer.result_summary.get("judge"),
            "prefill_count": len(sample_writer.prefills),
            "prefill_statuses": [prefill["status"] for prefill in sample_writer.prefills],
        }
        self.samples.append(sample_summary)
        self._write_manifest()

    def _write_manifest(self):
        manifest = {
            "run_dir": self.run_dir,
            "model_name": self.model_name,
            "result_path": self.out_file,
            "snapkv_topk_overlap_config": {
                "budget": self.config.budget,
                "window_size": self.config.window_size,
                "kernel_size": self.config.kernel_size,
                "max_prefill_tokens": self.config.max_prefill_tokens,
            },
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as fout:
            json.dump(manifest, fout, ensure_ascii=False, indent=2)


class SnapKVTopKOverlapSampleWriter:
    def __init__(self, run_writer, item, sample_index):
        self.run_writer = run_writer
        self.item = dict(item)
        self.sample_index = int(sample_index)
        base_name = sanitize_slug(
            self.item.get("_id", f"sample_{sample_index:04d}"),
            fallback=f"sample_{sample_index:04d}",
        )
        self.sample_id = f"sample_{sample_index:04d}_{base_name}"
        self.sample_dir = os.path.join(self.run_writer.samples_dir, self.sample_id)
        self.sample_rel_path = os.path.relpath(self.sample_dir, self.run_writer.run_dir)
        self.sample_json_path = os.path.join(self.sample_dir, "sample.json")
        self.prefills = []
        self.result_summary = {}
        os.makedirs(self.sample_dir, exist_ok=True)
        self._write_sample_json()

    def capture_prefill(self, model, tokenizer, prompt_text, inputs, label):
        prefill_index = len(self.prefills)
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        record = {
            "prefill_index": prefill_index,
            "label": label,
            "status": "pending",
            "prompt_text": prompt_text,
            "token_count": len(input_ids),
            "token_ids": input_ids,
            "tokens": build_token_entries(tokenizer, input_ids),
            "summary_file": None,
            "npz_file": None,
            "image_files": {},
            "valid_layer_indices": [],
        }
        self.prefills.append(record)
        self._write_sample_json()

        try:
            result = compute_snapkv_topk_overlap_observation(
                model=model,
                inputs=inputs,
                config=self.run_writer.config,
            )
            artifact_prefix = f"prefill_{prefill_index:03d}_snapkv_topk_overlap"
            paths = save_snapkv_topk_overlap_observation(
                result=result,
                output_dir=self.sample_dir,
                prefix=artifact_prefix,
            )
            record["status"] = result.summary.get("status", "saved")
            record["summary_file"] = os.path.basename(paths["summary"])
            record["npz_file"] = os.path.basename(paths["npz"])
            record["image_files"] = {
                name: os.path.basename(path)
                for name, path in paths.get("images", {}).items()
            }
            record["valid_layer_indices"] = result.summary.get("valid_layer_indices", [])
            record["valid_layer_count"] = len(record["valid_layer_indices"])
            if result.summary.get("reason"):
                record["reason"] = result.summary["reason"]
            if result.summary.get("error"):
                record["error"] = result.summary["error"]
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)

        self._write_sample_json()
        return record

    def finalize(self, item):
        self.result_summary = extract_result_summary(item)
        self._write_sample_json()
        self.run_writer.register_sample(self)

    def build_capture_status(self):
        keys = [
            "prefill_index",
            "label",
            "status",
            "token_count",
            "valid_layer_count",
            "summary_file",
            "npz_file",
            "image_files",
            "reason",
            "error",
        ]
        return [{key: record[key] for key in keys if key in record} for record in self.prefills]

    def _write_sample_json(self):
        sample_payload = {
            "sample_id": self.sample_id,
            "sample_index": self.sample_index,
            "item": extract_result_summary(self.item),
            "result": self.result_summary,
            "prefills": self.prefills,
        }
        with open(self.sample_json_path, "w", encoding="utf-8") as fout:
            json.dump(sample_payload, fout, ensure_ascii=False, indent=2)


def build_snapkv_observation_run_writer(args, out_file):
    if not args.snapkv_observation_mode:
        return None
    config = SnapKVObservationConfig(
        budget=args.snapkv_observation_budget,
        window_size=args.snapkv_observation_window_size,
        kernel_size=args.snapkv_observation_kernel_size,
        max_prefill_tokens=args.snapkv_observation_max_prefill_tokens,
        hidden_mix_profile_path=getattr(args, "hidden_mix_profile_path", None),
    )
    return SnapKVObservationRunWriter(
        root_dir=args.snapkv_observation_dir,
        model_name=args.model,
        out_file=out_file,
        config=config,
    )


def build_snapkv_topk_overlap_run_writer(args, out_file):
    if not args.snapkv_topk_overlap_mode:
        return None
    config = SnapKVObservationConfig(
        budget=args.snapkv_observation_budget,
        window_size=args.snapkv_observation_window_size,
        kernel_size=args.snapkv_observation_kernel_size,
        max_prefill_tokens=args.snapkv_observation_max_prefill_tokens,
        hidden_mix_profile_path=None,
    )
    return SnapKVTopKOverlapRunWriter(
        root_dir=args.snapkv_observation_dir,
        model_name=args.model,
        out_file=out_file,
        config=config,
    )


def validate_snapkv_observation_args(args):
    if args.snapkv_observation_budget < 1:
        raise ValueError("--snapkv_observation_budget must be at least 1.")
    if args.snapkv_observation_window_size < 1:
        raise ValueError("--snapkv_observation_window_size must be at least 1.")
    if args.snapkv_observation_kernel_size < 1:
        raise ValueError("--snapkv_observation_kernel_size must be at least 1.")
    if args.snapkv_observation_budget <= args.snapkv_observation_window_size:
        raise ValueError("--snapkv_observation_budget must be greater than --snapkv_observation_window_size.")
    if args.snapkv_observation_max_prefill_tokens is not None and args.snapkv_observation_max_prefill_tokens < 1:
        raise ValueError("--snapkv_observation_max_prefill_tokens must be at least 1 when provided.")
    if args.snapkv_observation_mode and args.n_proc != 1:
        raise ValueError("--snapkv_observation_mode currently requires --n_proc 1.")
    if args.snapkv_topk_overlap_mode and args.n_proc != 1:
        raise ValueError("--snapkv_topk_overlap_mode currently requires --n_proc 1.")


def add_snapkv_observation_args(parser):
    parser.add_argument("--snapkv_observation_mode", action="store_true")
    parser.add_argument(
        "--snapkv_topk_overlap_mode",
        action="store_true",
        help="Capture raw SnapKV per-layer topk index overlap and save a layer-by-layer heatmap.",
    )
    parser.add_argument(
        "--snapkv_observation_dir",
        type=str,
        default="output_dir/results_longbench/snapkv_observation",
    )
    parser.add_argument("--snapkv_observation_budget", type=int, default=4096)
    parser.add_argument("--snapkv_observation_window_size", type=int, default=8)
    parser.add_argument("--snapkv_observation_kernel_size", type=int, default=7)
    parser.add_argument(
        "--snapkv_observation_max_prefill_tokens",
        type=int,
        default=None,
        help="Skip SnapKV observation capture when the prefill token count exceeds this cap.",
    )
