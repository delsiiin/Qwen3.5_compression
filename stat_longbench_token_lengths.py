import argparse
import csv
import json
import os
import sys

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


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


def load_json(path):
    with open(path, encoding="utf-8") as fin:
        return json.load(fin)


def read_text(path):
    with open(path, encoding="utf-8") as fin:
        return fin.read()


def load_prompt_templates(prompt_dir="prompts"):
    return {
        "rag": read_text(os.path.join(prompt_dir, "0shot_rag.txt")),
        "no_context": read_text(os.path.join(prompt_dir, "0shot_no_context.txt")),
        "zero_shot": read_text(os.path.join(prompt_dir, "0shot.txt")),
        "cot": read_text(os.path.join(prompt_dir, "0shot_cot.txt")),
    }


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


def get_model_path(model_name):
    model_map = load_json("config/model2path.json")
    return model_map.get(model_name, model_name)


def get_max_input_len(model_maxlen, max_new_tokens):
    max_len = model_maxlen
    if max_len is None or max_len > 10**8:
        max_len = 120000
    return max(1, max_len - max_new_tokens)


def select_v1_datasets(dataset_args, use_longbench_e=False):
    defaults = LONG_BENCH_E_DATASETS if use_longbench_e else LONG_BENCH_DATASETS
    selected = parse_name_filter(dataset_args)
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
                item = json.loads(line)
                item["_id"] = str(item.get("_id") or f"{config_name}:{index}")
                item["dataset"] = dataset_name
                item["split"] = "test"
                item["longbench_e"] = bool(use_longbench_e)
                if "question" not in item and "input" in item:
                    item["question"] = item["input"]
                examples.append(item)
    return examples


def build_v1_prompt(prompt_format, item):
    return prompt_format.format(**item)


def build_v2_prompt(template, context, item):
    return (
        template.replace("$DOC$", context.strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )


def build_v2_example(item):
    example = {
        "_id": item["_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length": item["length"],
        "question": item["question"],
        "choice_A": item["choice_A"],
        "choice_B": item["choice_B"],
        "choice_C": item["choice_C"],
        "choice_D": item["choice_D"],
        "answer": item["answer"],
        "context": item["context"],
    }
    if "retrieved_context" in item:
        example["retrieved_context"] = item["retrieved_context"]
    return example


def load_longbench_v2(arrow_path=None, domains=None):
    if arrow_path:
        dataset = Dataset.from_file(arrow_path)
    else:
        dataset = load_dataset("THUDM/LongBench-v2", split="train")

    examples = [build_v2_example(item) for item in dataset]
    selected_domains = parse_name_filter(domains)
    if not selected_domains:
        return examples

    normalized = {domain.strip().lower() for domain in selected_domains}
    return [item for item in examples if item["domain"].strip().lower() in normalized]


def load_tokenizer(model_name, trust_remote_code=True, local_files_only=False):
    model_path = get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="left",
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model_path


def truncate_prompt_ids(prompt_ids, tokenizer, max_input_len):
    if len(prompt_ids) <= max_input_len:
        return prompt_ids, False
    half = max_input_len // 2
    truncated_ids = prompt_ids[:half] + prompt_ids[-(max_input_len - half) :]
    # Match run_longbench.py: it decodes the middle-truncated ids back to text
    # before feeding the prompt through tokenizer/chat template.
    truncated_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return tokenizer.encode(truncated_prompt, add_special_tokens=False), True


def count_prefill_tokens(prompt, tokenizer, max_new_tokens, model_maxlen, enable_thinking=False):
    raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_input_len = get_max_input_len(model_maxlen, max_new_tokens)

    if len(raw_prompt_ids) > max_input_len:
        half = max_input_len // 2
        truncated_ids = raw_prompt_ids[:half] + raw_prompt_ids[-(max_input_len - half) :]
        prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        truncated_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        was_truncated = True
    else:
        truncated_prompt_ids = raw_prompt_ids
        was_truncated = False

    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors=None,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors=None,
            )
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
    else:
        input_ids = tokenizer(prompt)["input_ids"]

    return {
        "prefill_tokens": len(input_ids),
        "raw_prompt_tokens": len(raw_prompt_ids),
        "truncated_prompt_tokens": len(truncated_prompt_ids),
        "was_truncated": was_truncated,
    }


def empty_record(source, group):
    return {
        "source": source,
        "group": group,
        "count": 0,
        "max_prefill_tokens": -1,
        "max_prefill_id": None,
        "max_raw_prompt_tokens": -1,
        "max_raw_prompt_id": None,
        "max_truncated_prompt_tokens": -1,
        "max_truncated_prompt_id": None,
        "truncated_count": 0,
    }


def update_record(record, item_id, lengths):
    record["count"] += 1
    if lengths["was_truncated"]:
        record["truncated_count"] += 1

    if lengths["prefill_tokens"] > record["max_prefill_tokens"]:
        record["max_prefill_tokens"] = lengths["prefill_tokens"]
        record["max_prefill_id"] = item_id

    if lengths["raw_prompt_tokens"] > record["max_raw_prompt_tokens"]:
        record["max_raw_prompt_tokens"] = lengths["raw_prompt_tokens"]
        record["max_raw_prompt_id"] = item_id

    if lengths["truncated_prompt_tokens"] > record["max_truncated_prompt_tokens"]:
        record["max_truncated_prompt_tokens"] = lengths["truncated_prompt_tokens"]
        record["max_truncated_prompt_id"] = item_id


def iter_with_progress(items, enabled, desc):
    if not enabled:
        return items
    from tqdm import tqdm

    return tqdm(items, desc=desc)


def stat_v1(args, tokenizer):
    dataset2prompt = load_json("config/dataset2prompt.json")
    dataset2maxlen = load_json("config/dataset2maxlen.json")
    datasets = select_v1_datasets(args.dataset, use_longbench_e=args.longbench_e)
    data = load_longbench_v1(datasets, use_longbench_e=args.longbench_e)
    records = {dataset: empty_record("run_longbench_v1.py", dataset) for dataset in datasets}

    for item in iter_with_progress(data, args.progress, "LongBench-v1"):
        dataset_name = item["dataset"]
        prompt = build_v1_prompt(dataset2prompt[dataset_name], item)
        lengths = count_prefill_tokens(
            prompt=prompt,
            tokenizer=tokenizer,
            max_new_tokens=dataset2maxlen[dataset_name],
            model_maxlen=args.model_maxlen,
            enable_thinking=args.enable_thinking,
        )
        update_record(records[dataset_name], item["_id"], lengths)

    return [records[dataset] for dataset in datasets]


def stat_v2(args, tokenizer):
    prompt_templates = load_prompt_templates()
    data = load_longbench_v2(arrow_path=args.longbench_v2_arrow, domains=args.domain)
    records = {}

    for item in iter_with_progress(data, args.progress, "LongBench-v2"):
        group = item[args.v2_group_by]
        record = records.setdefault(group, empty_record("run_longbench.py", group))
        context = item["context"]
        template_name = args.v2_prompt_mode

        if args.rag > 0:
            template_name = "rag"
            retrieved = item.get("retrieved_context", [])[: args.rag]
            retrieved = sorted(retrieved, key=lambda chunk: chunk["c_idx"])
            context = "\n\n".join(
                f"Retrieved chunk {idx + 1}: {chunk['content']}"
                for idx, chunk in enumerate(retrieved)
            )

        prompt = build_v2_prompt(prompt_templates[template_name], context, item)
        max_new_tokens = 1024 if template_name == "cot" else 128
        lengths = count_prefill_tokens(
            prompt=prompt,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            model_maxlen=args.model_maxlen,
            enable_thinking=template_name == "cot",
        )
        update_record(record, item["_id"], lengths)

    return [records[key] for key in sorted(records)]


def write_json(records, output_file):
    payload = {"records": records}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as fout:
            fout.write(text + "\n")
    else:
        print(text)


def write_csv(records, output_file):
    fieldnames = [
        "model",
        "model_path",
        "model_maxlen",
        "source",
        "group",
        "count",
        "max_prefill_tokens",
        "max_prefill_id",
        "max_raw_prompt_tokens",
        "max_raw_prompt_id",
        "max_truncated_prompt_tokens",
        "max_truncated_prompt_id",
        "truncated_count",
    ]
    if output_file:
        fout = open(output_file, "w", encoding="utf-8", newline="")
        should_close = True
    else:
        fout = sys.stdout
        should_close = False
    try:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    finally:
        if should_close:
            fout.close()


def write_table(records, output_file):
    headers = [
        "source",
        "group",
        "count",
        "max_prefill_tokens",
        "max_prefill_id",
        "max_raw_prompt_tokens",
        "truncated_count",
    ]
    rows = [[str(record[key]) for key in headers] for record in records]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(header)
        for idx, header in enumerate(headers)
    ]
    lines = [
        "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers)))
        for row in rows
    )
    text = "\n".join(lines)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as fout:
            fout.write(text + "\n")
    else:
        print(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count max tokenized input lengths for run_longbench.py and "
            "run_longbench_v1.py without loading the model."
        )
    )
    parser.add_argument("--suite", choices=["v1", "v2", "both"], default="both")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--model_maxlen", type=int, default=120000)
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the existing scripts' tokenizer loading behavior.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use only local HuggingFace cache/model files.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        action="append",
        default=None,
        help="LongBench-v1 dataset filter. Repeat or use comma-separated names.",
    )
    parser.add_argument("--longbench_e", action="store_true", help="Use LongBench-E files for v1.")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Pass enable_thinking=True when applying v1 chat templates.",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        help="LongBench-v2 domain filter. Repeat or use comma-separated names.",
    )
    parser.add_argument(
        "--v2_group_by",
        choices=["domain", "sub_domain", "difficulty"],
        default="domain",
    )
    parser.add_argument(
        "--v2_prompt_mode",
        choices=["zero_shot", "no_context", "cot"],
        default="zero_shot",
        help="Prompt branch to mirror from run_longbench.py when --rag is 0.",
    )
    parser.add_argument("--rag", type=int, default=0)
    parser.add_argument(
        "--longbench_v2_arrow",
        type=str,
        default=None,
        help="Optional local json-train.arrow path for offline LongBench-v2 loading.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer, model_path = load_tokenizer(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    records = []
    if args.suite in {"v1", "both"}:
        records.extend(stat_v1(args, tokenizer))
    if args.suite in {"v2", "both"}:
        records.extend(stat_v2(args, tokenizer))

    for record in records:
        record["model"] = args.model
        record["model_path"] = model_path
        record["model_maxlen"] = args.model_maxlen

    if args.format == "json":
        write_json(records, args.output)
    elif args.format == "csv":
        write_csv(records, args.output)
    else:
        write_table(records, args.output)


if __name__ == "__main__":
    main()
