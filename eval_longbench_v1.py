import argparse
import json
import os
from collections import defaultdict

import numpy as np

from metrics_longbench_v1 import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)


RESULTS_DIR = "output_dir/results_longbench_v1"
E_BUCKETS = ("0-4k", "4-8k", "8k+")

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--model", type=str, default=None, help="Compatibility filter: only evaluate run files containing this text.")
    parser.add_argument("--run", type=str, default=None, help="Only evaluate run files containing this text.")
    parser.add_argument("--e", action="store_true", help="Evaluate LongBench-E records and report length buckets.")
    return parser.parse_args(args)


def normalize_prediction(dataset, prediction):
    prediction = "" if prediction is None else str(prediction)
    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        prediction = prediction.lstrip("\n").split("\n")[0]
    return prediction


def score_one(dataset, prediction, ground_truths, all_classes):
    metric = dataset2metric.get(dataset)
    if metric is None:
        return None

    prediction = normalize_prediction(dataset, prediction)
    score = 0.0
    for ground_truth in ground_truths or []:
        score = max(score, metric(prediction, ground_truth, all_classes=all_classes or []))
    return score


def scorer(dataset, rows):
    if not rows:
        return 0.0

    scores = []
    for row in rows:
        score = score_one(
            dataset=dataset,
            prediction=row.get("pred"),
            ground_truths=row.get("answers"),
            all_classes=row.get("all_classes"),
        )
        if score is not None:
            scores.append(score)
    return round(100 * float(np.mean(scores)), 2) if scores else 0.0


def length_bucket(length):
    try:
        length = int(length)
    except (TypeError, ValueError):
        return "8k+"
    if length < 4000:
        return "0-4k"
    if length < 8000:
        return "4-8k"
    return "8k+"


def scorer_e(dataset, rows):
    scores = {bucket: [] for bucket in E_BUCKETS}
    for row in rows:
        score = score_one(
            dataset=dataset,
            prediction=row.get("pred"),
            ground_truths=row.get("answers"),
            all_classes=row.get("all_classes"),
        )
        if score is None:
            continue
        scores[length_bucket(row.get("length"))].append(score)
    return {
        bucket: round(100 * float(np.mean(values)), 2) if values else 0.0
        for bucket, values in scores.items()
    }


def should_evaluate_file(filename, args):
    if not filename.endswith(".jsonl"):
        return False
    if args.model and args.model not in filename:
        return False
    if args.run and args.run not in filename:
        return False
    return True


def iter_result_rows(path, evaluate_e):
    with open(path, encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed JSON in {path}:{line_number}: {exc}")
                continue
            if row.get("longbench_e", False) != evaluate_e:
                continue
            if not row.get("dataset") or "answers" not in row:
                print(f"Skipping incomplete row in {path}:{line_number}")
                continue
            yield row


def evaluate_run(path, evaluate_e):
    rows_by_dataset = defaultdict(list)
    for row in iter_result_rows(path, evaluate_e):
        rows_by_dataset[row["dataset"]].append(row)

    if not rows_by_dataset:
        return None

    scores = {}
    for dataset in sorted(rows_by_dataset):
        rows = rows_by_dataset[dataset]
        if evaluate_e:
            scores[dataset] = scorer_e(dataset, rows)
        else:
            scores[dataset] = scorer(dataset, rows)
    return scores


def main(args):
    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    result = {}
    files = sorted(
        filename for filename in os.listdir(args.results_dir)
        if should_evaluate_file(filename, args)
    )
    print("Evaluating on:", files)

    for filename in files:
        path = os.path.join(args.results_dir, filename)
        if not os.path.isfile(path):
            continue
        scores = evaluate_run(path, evaluate_e=args.e)
        if scores is not None:
            result[os.path.splitext(filename)[0]] = scores

    output_name = "result_e.json" if args.e else "result.json"
    out_path = os.path.join(args.results_dir, output_name)
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=4)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main(parse_args())
