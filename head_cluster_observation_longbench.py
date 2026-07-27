import json
import os

from attn_heatmap import (
    build_run_dir,
    build_token_entries,
    extract_result_summary,
    sanitize_slug,
)
from models.compression.experiments.head_cluster_observation import (
    HEAD_BUDGET_ATTENTION_SUBMODE,
    HEAD_CLUSTER_PCA_METHODS,
    HEAD_CLUSTER_PCA_SUBMODE,
    HEAD_CLUSTER_TOKEN_DISTRIBUTION_SUBMODE,
    SUPPORTED_HEAD_CLUSTER_OBSERVATION_METHODS,
    SUPPORTED_HEAD_CLUSTER_OBSERVATION_SUBMODES,
    HeadClusterObservationConfig,
    compute_head_cluster_observation,
    save_head_cluster_observation,
)


class HeadClusterObservationRunWriter:
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
        return HeadClusterObservationSampleWriter(self, item, sample_index)

    def register_sample(self, sample_writer):
        self.samples.append(
            {
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
        )
        self._write_manifest()

    def _write_manifest(self):
        payload = {
            "run_dir": self.run_dir,
            "model_name": self.model_name,
            "result_path": self.out_file,
            "head_cluster_observation_config": {
                "submode": self.config.submode,
                "max_prefill_tokens": self.config.max_prefill_tokens,
            },
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as fout:
            json.dump(payload, fout, ensure_ascii=False, indent=2)


class HeadClusterObservationSampleWriter:
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
            result = compute_head_cluster_observation(
                model=model,
                inputs=inputs,
                config=self.run_writer.config,
            )
            artifact_prefix = f"prefill_{prefill_index:03d}_head_cluster_observation"
            paths = save_head_cluster_observation(
                result=result,
                output_dir=self.sample_dir,
                prefix=artifact_prefix,
            )
            record["status"] = result.summary.get("status", "saved")
            record["summary_file"] = os.path.basename(paths["summary"])
            record["npz_file"] = os.path.basename(paths["npz"])
            record["image_files"] = {
                name: os.path.basename(path) for name, path in paths.get("images", {}).items()
            }
            record["valid_layer_indices"] = result.summary.get("valid_layer_indices", [])
            record["valid_layer_count"] = len(record["valid_layer_indices"])
            for key in ("reason", "error", "compression_method", "compression_config"):
                if result.summary.get(key) is not None:
                    record[key] = result.summary[key]
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
            "compression_method",
            "compression_config",
            "valid_layer_count",
            "summary_file",
            "npz_file",
            "image_files",
            "reason",
            "error",
        ]
        return [{key: record[key] for key in keys if key in record} for record in self.prefills]

    def _write_sample_json(self):
        payload = {
            "sample_id": self.sample_id,
            "sample_index": self.sample_index,
            "item": extract_result_summary(self.item),
            "result": self.result_summary,
            "prefills": self.prefills,
        }
        with open(self.sample_json_path, "w", encoding="utf-8") as fout:
            json.dump(payload, fout, ensure_ascii=False, indent=2)


def build_head_cluster_observation_run_writer(args, out_file):
    if not args.head_cluster_observation_mode:
        return None
    return HeadClusterObservationRunWriter(
        root_dir=args.head_cluster_observation_dir,
        model_name=args.model,
        out_file=out_file,
        config=HeadClusterObservationConfig(
            submode=args.head_cluster_observation_submode,
            max_prefill_tokens=args.head_cluster_observation_max_prefill_tokens,
        ),
    )


def validate_head_cluster_observation_args(args):
    if args.head_cluster_observation_submode not in SUPPORTED_HEAD_CLUSTER_OBSERVATION_SUBMODES:
        raise ValueError(
            "--head_cluster_observation_submode must be one of "
            f"{sorted(SUPPORTED_HEAD_CLUSTER_OBSERVATION_SUBMODES)}."
        )
    if (
        args.head_cluster_observation_max_prefill_tokens is not None
        and args.head_cluster_observation_max_prefill_tokens < 1
    ):
        raise ValueError(
            "--head_cluster_observation_max_prefill_tokens must be at least 1 when provided."
        )
    if not args.head_cluster_observation_mode:
        return
    if not args.compression:
        raise ValueError("--head_cluster_observation_mode requires --compression.")
    supported_methods = (
        HEAD_CLUSTER_PCA_METHODS
        if args.head_cluster_observation_submode == HEAD_CLUSTER_PCA_SUBMODE
        else SUPPORTED_HEAD_CLUSTER_OBSERVATION_METHODS
    )
    if args.compression_mode not in supported_methods:
        raise ValueError(
            f"--head_cluster_observation_submode {args.head_cluster_observation_submode!r} "
            "supports --compression_mode values "
            f"{sorted(supported_methods)}."
        )
    if args.n_proc != 1:
        raise ValueError("--head_cluster_observation_mode currently requires --n_proc 1.")


def add_head_cluster_observation_args(parser):
    parser.add_argument(
        "--head_cluster_observation_mode",
        action="store_true",
        help=(
            "Run an isolated cached prefill with the active compression method and save real "
            "per-layer KV-head budget/raw-attention, head-cluster/PCA, or cluster token-allocation "
            "artifacts."
        ),
    )
    parser.add_argument(
        "--head_cluster_observation_submode",
        type=str,
        choices=sorted(SUPPORTED_HEAD_CLUSTER_OBSERVATION_SUBMODES),
        default=HEAD_BUDGET_ATTENTION_SUBMODE,
        help=(
            "Head-cluster observation experiment: head_budget_attention saves existing budget "
            "and attention plots; head_cluster_pca saves cluster partitions and per-KV-head "
            "raw-attention PCA plots; "
            f"{HEAD_CLUSTER_TOKEN_DISTRIBUTION_SUBMODE} saves each layer's share of selected "
            "historical (KV head, token) slots by raw-attention head cluster."
        ),
    )
    parser.add_argument(
        "--head_cluster_observation_dir",
        type=str,
        default="output_dir/results_longbench/head_cluster_observation",
    )
    parser.add_argument(
        "--head_cluster_observation_max_prefill_tokens",
        type=int,
        default=None,
        help="Skip head-cluster observation capture when the prefill token count exceeds this cap.",
    )
