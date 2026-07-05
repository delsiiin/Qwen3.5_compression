import argparse
import json
import os
import random
import re
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from attn_heatmap import is_qwen_attn_heatmap_model
from misc import load_json, select_unprocessed
from query_window_similarity import (
    SIMILARITY_STATE_HIDDEN,
    SUPPORTED_SIMILARITY_STATES,
)
from run_longbench import (
    build_attn_output_ratio_run_writer,
    build_attn_run_writer,
    build_hidden_state_pca_run_writer,
    build_query_window_similarity_run_writer,
    load_model_and_tokenizer,
    query_llm,
)
from snapkv_observation_longbench import (
    add_snapkv_observation_args,
    build_snapkv_observation_run_writer,
    validate_snapkv_observation_args,
)


LONG_BENCH_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

LONG_BENCH_E_DATASETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

LONG_BENCH_V1_DATA_DIR = os.path.join("data", "LongBench_v1")


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_name_filter(name_args):
    if not name_args:
        return None
    names = []
    for name_arg in name_args:
        for name in name_arg.split(","):
            name = name.strip()
            if name and name not in names:
                names.append(name)
    return names or None


def sanitize_slug(value, fallback="run"):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return value or fallback


def get_dataset_suffix(datasets):
    if not datasets:
        return ""
    return "_datasets_" + "_".join(sanitize_slug(dataset) for dataset in datasets)


def build_output_path(args, datasets):
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        return args.output_file

    os.makedirs(args.save_dir, exist_ok=True)
    output_prefix = sanitize_slug(args.model.split("/")[-1])
    output_prefix += "_longbench_e" if args.e else "_longbench"
    output_prefix += get_dataset_suffix(datasets)

    if args.compression:
        compression_mode = args.compression_mode or "compressed"
        output_prefix = f"{output_prefix}_{compression_mode}_budget_{args.compression_budget}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(args.save_dir, f"{output_prefix}_{timestamp}.jsonl")


def select_datasets(args):
    defaults = LONG_BENCH_E_DATASETS if args.e else LONG_BENCH_DATASETS
    selected = parse_name_filter(args.dataset)
    if not selected:
        return defaults

    available = set(defaults)
    missing = [dataset for dataset in selected if dataset not in available]
    if missing:
        raise ValueError(
            f"Dataset(s) not available for this split: {', '.join(missing)}. "
            f"Available datasets: {', '.join(defaults)}"
        )
    return selected


def load_longbench_v1(datasets, use_longbench_e=False):
    examples = []
    for dataset_name in datasets:
        config_name = f"{dataset_name}_e" if use_longbench_e else dataset_name
        data_file = os.path.join(LONG_BENCH_V1_DATA_DIR, f"{config_name}.jsonl")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"LongBench v1 data file not found: {data_file}")

        with open(data_file, encoding="utf-8") as fin:
            for index, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in {data_file}:{index + 1}: {exc}"
                    ) from exc

                item["_id"] = str(item.get("_id") or f"{config_name}:{index}")
                item["dataset"] = dataset_name
                item["split"] = "test"
                item["longbench_e"] = bool(use_longbench_e)
                if "question" not in item and "input" in item:
                    item["question"] = item["input"]
                examples.append(item)
    return examples


def load_processed_ids_v1(out_file):
    if not os.path.exists(out_file):
        return set()
    processed_ids = set()
    with open(out_file, encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed JSON in {out_file}:{line_number}: {exc}")
                continue
            if "_id" in item:
                processed_ids.add(item["_id"])
    return processed_ids


def build_prompt(prompt_format, item):
    return prompt_format.format(**item)


def build_result_item(item, pred, prompt, capture_fields=None):
    result = {
        "_id": item["_id"],
        "dataset": item["dataset"],
        "split": item["split"],
        "longbench_e": item["longbench_e"],
        "pred": pred,
        "answers": item["answers"],
        "all_classes": item.get("all_classes", []),
        "length": item.get("length"),
        "input": item.get("input"),
        "question": item.get("question"),
    }
    if capture_fields:
        result.update(capture_fields)
    return result


def finalize_sample_writers(item, writers):
    for writer in writers:
        if writer is not None:
            writer.finalize(item)


def build_capture_fields(args, sample_writers):
    fields = {}
    (
        attn_sample_writer,
        query_window_sample_writer,
        attn_output_ratio_sample_writer,
        snapkv_observation_sample_writer,
        hidden_state_pca_sample_writer,
    ) = sample_writers

    if attn_sample_writer is not None:
        fields["attn_capture_status"] = attn_sample_writer.build_capture_status()
        fields["attn_artifact"] = os.path.relpath(
            attn_sample_writer.sample_dir,
            start=args.attn_heatmap_dir,
        )
    if query_window_sample_writer is not None:
        fields["query_window_similarity_status"] = query_window_sample_writer.build_capture_status()
        fields["query_window_similarity_artifact"] = os.path.relpath(
            query_window_sample_writer.sample_dir,
            start=args.query_window_similarity_dir,
        )
    if attn_output_ratio_sample_writer is not None:
        fields["attn_output_ratio_status"] = attn_output_ratio_sample_writer.build_capture_status()
        fields["attn_output_ratio_artifact"] = os.path.relpath(
            attn_output_ratio_sample_writer.sample_dir,
            start=args.attn_output_ratio_dir,
        )
    if snapkv_observation_sample_writer is not None:
        fields["snapkv_observation_status"] = snapkv_observation_sample_writer.build_capture_status()
        fields["snapkv_observation_artifact"] = os.path.relpath(
            snapkv_observation_sample_writer.sample_dir,
            start=args.snapkv_observation_dir,
        )
    if hidden_state_pca_sample_writer is not None:
        fields["hidden_state_pca_status"] = hidden_state_pca_sample_writer.build_capture_status()
        fields["hidden_state_pca_artifact"] = os.path.relpath(
            hidden_state_pca_sample_writer.sample_dir,
            start=args.hidden_state_pca_dir,
        )
    return fields


def write_jsonl(out_file, item):
    with open(out_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        fout.flush()


def get_pred(data, args, out_file):
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        attn_heatmap_mode=args.attn_heatmap_mode,
        compression=args.compression,
        compression_mode=args.compression_mode,
        compression_budget=args.compression_budget,
        hidden_mix_profile_path=args.hidden_mix_profile_path,
        group_threshold_ema_decay=args.group_threshold_ema_decay,
    )
    dataset2prompt = load_json("config/dataset2prompt.json")
    dataset2maxlen = load_json("config/dataset2maxlen.json")

    attn_run_writer = build_attn_run_writer(args, out_file, model)
    query_window_run_writer = build_query_window_similarity_run_writer(args, out_file)
    attn_output_ratio_run_writer = build_attn_output_ratio_run_writer(args, out_file)
    snapkv_observation_run_writer = build_snapkv_observation_run_writer(args, out_file)
    hidden_state_pca_run_writer = build_hidden_state_pca_run_writer(args, out_file)

    for item in tqdm(data):
        item = dict(item)
        attn_sample_writer = attn_run_writer.new_sample(item) if attn_run_writer is not None else None
        query_window_sample_writer = (
            query_window_run_writer.new_sample(item)
            if query_window_run_writer is not None
            else None
        )
        attn_output_ratio_sample_writer = (
            attn_output_ratio_run_writer.new_sample(item)
            if attn_output_ratio_run_writer is not None
            else None
        )
        snapkv_observation_sample_writer = (
            snapkv_observation_run_writer.new_sample(item)
            if snapkv_observation_run_writer is not None
            else None
        )
        hidden_state_pca_sample_writer = (
            hidden_state_pca_run_writer.new_sample(item)
            if hidden_state_pca_run_writer is not None
            else None
        )
        sample_writers = (
            attn_sample_writer,
            query_window_sample_writer,
            attn_output_ratio_sample_writer,
            snapkv_observation_sample_writer,
            hidden_state_pca_sample_writer,
        )

        try:
            dataset_name = item["dataset"]
            prompt = build_prompt(dataset2prompt[dataset_name], item)
            pred, _ = query_llm(
                prompt,
                args.model,
                model,
                tokenizer,
                args.model_maxlen,
                temperature=0,
                max_new_tokens=dataset2maxlen[dataset_name],
                enable_thinking=args.enable_thinking,
                attn_sample_writer=attn_sample_writer,
                attn_output_ratio_sample_writer=attn_output_ratio_sample_writer,
                query_window_sample_writer=query_window_sample_writer,
                snapkv_observation_sample_writer=snapkv_observation_sample_writer,
                hidden_state_pca_sample_writer=hidden_state_pca_sample_writer,
                prefill_label=dataset_name,
            )
            result_item = build_result_item(
                item=item,
                pred=pred,
                prompt=prompt,
                capture_fields=build_capture_fields(args, sample_writers),
            )
            finalize_sample_writers(result_item, sample_writers)
            write_jsonl(out_file, result_item)
        except Exception as exc:
            print(f"Skipping sample {item.get('_id', 'unknown')} due to error: {exc}")
            traceback.print_exc()
            error_item = build_result_item(
                item=item,
                pred=None,
                prompt="",
                capture_fields=build_capture_fields(args, sample_writers),
            )
            error_item["error"] = str(exc)
            finalize_sample_writers(error_item, sample_writers)
            write_jsonl(out_file, error_item)


def validate_args(args):
    if args.model_maxlen < 1:
        raise ValueError("--model_maxlen must be at least 1.")
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("--num_samples must be at least 1 when provided.")
    if args.n_proc < 1:
        raise ValueError("--n_proc must be at least 1.")
    if args.compression and not args.compression_mode:
        raise ValueError("--compression requires --compression_mode.")
    if args.compression and args.compression_budget < 1:
        raise ValueError("--compression_budget must be at least 1 when compression is enabled.")
    if args.query_window_size < 1:
        raise ValueError("--query_window_size must be at least 1.")
    if args.query_window_max_prefill_tokens is not None and args.query_window_max_prefill_tokens < 1:
        raise ValueError("--query_window_max_prefill_tokens must be at least 1 when provided.")
    if args.attn_output_ratio_max_prefill_tokens is not None and args.attn_output_ratio_max_prefill_tokens < 1:
        raise ValueError("--attn_output_ratio_max_prefill_tokens must be at least 1 when provided.")
    if args.hidden_state_pca_max_prefill_tokens is not None and args.hidden_state_pca_max_prefill_tokens < 1:
        raise ValueError("--hidden_state_pca_max_prefill_tokens must be at least 1 when provided.")
    if (
        args.hidden_state_pca_token_end is not None
        and args.hidden_state_pca_token_start >= 0
        and args.hidden_state_pca_token_end >= 0
        and args.hidden_state_pca_token_end <= args.hidden_state_pca_token_start
    ):
        raise ValueError("--hidden_state_pca_token_end must be greater than --hidden_state_pca_token_start.")
    if args.query_window_similarity_submode not in SUPPORTED_SIMILARITY_STATES:
        raise ValueError(
            "--query_window_similarity_submode must be one of "
            f"{sorted(SUPPORTED_SIMILARITY_STATES)}."
        )
    if args.attn_heatmap_mode:
        if not is_qwen_attn_heatmap_model(args.model):
            raise ValueError("--attn_heatmap_mode currently supports only qwen3.5-* models in this repository.")
        if args.n_proc != 1:
            raise ValueError("--attn_heatmap_mode currently requires --n_proc 1.")
    if args.query_window_similarity_mode and args.n_proc != 1:
        raise ValueError("--query_window_similarity_mode currently requires --n_proc 1.")
    if args.attn_output_ratio_mode and args.n_proc != 1:
        raise ValueError("--attn_output_ratio_mode currently requires --n_proc 1.")
    if args.hidden_state_pca_mode and args.n_proc != 1:
        raise ValueError("--hidden_state_pca_mode currently requires --n_proc 1.")
    validate_snapkv_observation_args(args)


def main(args):
    seed_everything(args.seed)
    validate_args(args)

    datasets = select_datasets(args)
    out_file = build_output_path(args, datasets if args.dataset else None)
    print(args)
    print(f"Writing results to {out_file}")

    data_all = load_longbench_v1(datasets, use_longbench_e=args.e)
    processed_ids = load_processed_ids_v1(out_file)
    data = select_unprocessed(data_all, processed_ids)
    if args.num_samples is not None:
        data = data[:args.num_samples]
        print(f"Limited this run to {len(data)} example(s).")

    if len(data) == 0:
        print("No new examples to process.")
        return

    if args.n_proc == 1:
        get_pred(data, args, out_file)
    else:
        print("Warning: each process will load its own transformers model copy.")
        data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
        processes = []
        for rank in range(args.n_proc):
            process = mp.Process(target=get_pred, args=(data_subsets[rank], args, out_file))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Worker process exited with code {process.exitcode}.")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="output_dir/results_longbench_v1")
    parser.add_argument("--output_file", type=str, default=None, help="Append to this JSONL file and skip existing _id values.")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--model_maxlen", type=int, default=120000, help="Model context length used for prompt truncation.")
    parser.add_argument("--dataset", "-d", action="append", default=None, help="Only run selected LongBench v1 dataset(s). Repeat or use comma-separated names.")
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E.")
    parser.add_argument("--num_samples", "--max_samples", type=int, default=None, help="Only run the first N unprocessed examples.")
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action="store_true", help="Pass enable_thinking=True to chat templates that support it.")
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--compression_mode", type=str, default=None)
    parser.add_argument("--compression_budget", type=int, default=4096)
    parser.add_argument("--hidden_mix_profile_path", type=str, default=None)
    parser.add_argument("--group_threshold_ema_decay", type=float, default=None)
    parser.add_argument("--attn_heatmap_mode", action="store_true")
    parser.add_argument("--attn_heatmap_dir", type=str, default="output_dir/results_longbench_v1/attn_heatmaps")
    parser.add_argument("--attn_max_prefill_tokens", type=int, default=None, help="Skip attention heatmap capture when the prefill token count exceeds this cap.")
    parser.add_argument("--query_window_similarity_mode", action="store_true")
    parser.add_argument("--query_window_similarity_dir", type=str, default="output_dir/results_longbench_v1/query_window_similarity")
    parser.add_argument("--query_window_size", type=int, default=8, help="Number of prompt-tail tokens used for layer-wise query-window analysis.")
    parser.add_argument(
        "--query_window_similarity_submode",
        "--query_window_similarity_state",
        dest="query_window_similarity_submode",
        type=str,
        choices=sorted(SUPPORTED_SIMILARITY_STATES),
        default=SIMILARITY_STATE_HIDDEN,
    )
    parser.add_argument("--query_window_max_prefill_tokens", type=int, default=None, help="Skip query window similarity capture when the prefill token count exceeds this cap.")
    parser.add_argument("--attn_output_ratio_mode", action="store_true")
    parser.add_argument("--attn_output_ratio_dir", type=str, default="output_dir/results_longbench_v1/attn_output_ratios")
    parser.add_argument("--attn_output_ratio_layers", type=str, default="all")
    parser.add_argument("--attn_output_ratio_max_prefill_tokens", type=int, default=None, help="Skip attention-output ratio capture when the prefill token count exceeds this cap.")
    parser.add_argument("--hidden_state_pca_mode", action="store_true")
    parser.add_argument("--hidden_state_pca_dir", type=str, default="output_dir/results_longbench_v1/hidden_state_pca")
    parser.add_argument("--hidden_state_pca_layers", type=str, default="all")
    parser.add_argument("--hidden_state_pca_token_start", type=int, default=0)
    parser.add_argument("--hidden_state_pca_token_end", type=int, default=None)
    parser.add_argument("--hidden_state_pca_max_prefill_tokens", type=int, default=None)
    add_snapkv_observation_args(parser)
    parsed = parser.parse_args(args)
    if parsed.snapkv_observation_dir == "output_dir/results_longbench/snapkv_observation":
        parsed.snapkv_observation_dir = "output_dir/results_longbench_v1/snapkv_observation"
    return parsed


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(parse_args())
