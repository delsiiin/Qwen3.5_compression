import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def is_qwen_attn_heatmap_model(model_name):
    return "qwen3.5" in model_name.lower()


def get_text_config(model):
    return getattr(model.config, "text_config", model.config)


def get_full_attention_layer_indices(model):
    text_config = get_text_config(model)
    return [
        layer_idx
        for layer_idx, layer_type in enumerate(getattr(text_config, "layer_types", []))
        if layer_type == "full_attention"
    ]


def sanitize_slug(value, fallback="item"):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return value or fallback


def build_run_dir(base_dir, out_file):
    run_name = os.path.splitext(os.path.basename(out_file))[0]
    return os.path.join(base_dir, sanitize_slug(run_name, fallback="run"))


def extract_result_summary(item):
    fields = [
        "_id",
        "domain",
        "sub_domain",
        "difficulty",
        "length",
        "question",
        "answer",
        "pred",
        "judge",
        "response",
        "response_cot",
    ]
    return {field: item[field] for field in fields if field in item}


def build_token_entries(tokenizer, token_ids):
    token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
    token_texts = []
    try:
        token_texts = tokenizer.batch_decode(
            [[token_id] for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        token_texts = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]

    return [
        {
            "index": idx,
            "id": int(token_id),
            "piece": token_piece,
            "text": token_text,
        }
        for idx, (token_id, token_piece, token_text) in enumerate(zip(token_ids, token_pieces, token_texts))
    ]


@dataclass
class PrefillCaptureResult:
    record: dict[str, Any]


class AttentionCaptureRecorder:
    def __init__(self, expected_layers):
        self.expected_layers = [int(layer_idx) for layer_idx in expected_layers]
        self._captured = {}

    def record_attention(self, layer_idx, attn_weights):
        if attn_weights is None or attn_weights.ndim != 4:
            return
        if attn_weights.shape[0] != 1 or attn_weights.shape[2] <= 1:
            return
        self._captured[int(layer_idx)] = (
            attn_weights[0].detach().to(dtype=torch.float16).cpu().contiguous().numpy()
        )

    def export(self):
        captured_layers = [layer_idx for layer_idx in self.expected_layers if layer_idx in self._captured]
        if not captured_layers:
            raise ValueError("No full-attention layers were captured during prefill.")
        attn = np.stack([self._captured[layer_idx] for layer_idx in captured_layers], axis=0)
        missing_layers = [layer_idx for layer_idx in self.expected_layers if layer_idx not in self._captured]
        return attn, captured_layers, missing_layers


class AttentionHeatmapRunWriter:
    def __init__(self, root_dir, model_name, out_file, full_attention_layers, max_prefill_tokens):
        self.root_dir = root_dir
        self.model_name = model_name
        self.out_file = os.path.abspath(out_file)
        self.full_attention_layers = [int(layer_idx) for layer_idx in full_attention_layers]
        self.max_prefill_tokens = int(max_prefill_tokens) if max_prefill_tokens is not None else None
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
        return AttentionHeatmapSampleWriter(self, item, sample_index)

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
            "full_attention_layers": self.full_attention_layers,
            "attn_max_prefill_tokens": self.max_prefill_tokens,
            "attn_prefill_cap_mode": "fixed" if self.max_prefill_tokens is not None else "none",
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as fout:
            json.dump(manifest, fout, ensure_ascii=False, indent=2)


class AttentionHeatmapSampleWriter:
    def __init__(self, run_writer, item, sample_index):
        self.run_writer = run_writer
        self.item = dict(item)
        self.sample_index = int(sample_index)
        base_name = sanitize_slug(self.item.get("_id", f"sample_{sample_index:04d}"), fallback=f"sample_{sample_index:04d}")
        self.sample_id = f"sample_{sample_index:04d}_{base_name}"
        self.sample_dir = os.path.join(self.run_writer.samples_dir, self.sample_id)
        self.sample_rel_path = os.path.relpath(self.sample_dir, self.run_writer.run_dir)
        self.sample_json_path = os.path.join(self.sample_dir, "sample.json")
        self.prefills = []
        self.result_summary = {}
        os.makedirs(self.sample_dir, exist_ok=True)
        self._write_sample_json()

    def capture_prefill(self, model, tokenizer, prompt_text, inputs, label, max_prefill_tokens=None):
        prefill_index = len(self.prefills)
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        prefill_cap = (
            int(max_prefill_tokens)
            if max_prefill_tokens is not None
            else self.run_writer.max_prefill_tokens
        )
        record = {
            "prefill_index": prefill_index,
            "label": label,
            "status": "pending",
            "prompt_text": prompt_text,
            "max_prefill_tokens": prefill_cap,
            "token_count": len(input_ids),
            "token_ids": input_ids,
            "tokens": build_token_entries(tokenizer, input_ids),
            "attention_file": None,
            "attn_shape": None,
            "layer_indices": [],
            "missing_layers": [],
        }
        self.prefills.append(record)
        self._write_sample_json()

        if prefill_cap is not None and len(input_ids) > prefill_cap: 
            record["status"] = "skipped_over_cap"
            record["reason"] = (
                f"Prompt token count {len(input_ids)} exceeds cap {prefill_cap}."
            )
            self._write_sample_json()
            return PrefillCaptureResult(record)

        recorder = AttentionCaptureRecorder(self.run_writer.full_attention_layers)
        try:
            with torch.inference_mode():
                model(**inputs, use_cache=False, attn_recorder=recorder)
            attn, layer_indices, missing_layers = recorder.export()
            attn_file_name = f"prefill_{prefill_index:03d}.npz"
            attn_path = os.path.join(self.sample_dir, attn_file_name)
            np.savez_compressed(
                attn_path,
                attn=attn.astype(np.float16, copy=False),
                layer_indices=np.asarray(layer_indices, dtype=np.int16),
            )
            record["status"] = "saved"
            record["attention_file"] = attn_file_name
            record["attn_shape"] = list(attn.shape)
            record["layer_indices"] = layer_indices
            record["missing_layers"] = missing_layers
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)

        self._write_sample_json()
        return PrefillCaptureResult(record)

    def finalize(self, item):
        self.result_summary = extract_result_summary(item)
        self._write_sample_json()
        self.run_writer.register_sample(self)

    def build_capture_status(self):
        keys = ["prefill_index", "label", "status", "token_count", "attn_shape", "reason", "error"]
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
