import argparse
import json
import os
import re
import traceback

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from attn_heatmap import (
    AttentionHeatmapRunWriter,
    get_full_attention_layer_indices,
    is_qwen_attn_heatmap_model,
)
from attn_output_ratio import AttnOutputRatioRunWriter
from hidden_state_pca_observation import (
    HiddenStatePCARunWriter,
    PCA_SUBMODE_KEY_VALUE_STATES,
    SUPPORTED_HIDDEN_STATE_PCA_SUBMODES,
)
from misc import (
    build_output_path,
    load_json,
    load_longbench_v2,
    load_processed_ids,
    load_prompt_templates,
    parse_domain_filter,
    select_unprocessed,
)
from query_window_similarity import (
    QueryWindowSimilarityRunWriter,
    SUPPORTED_SIMILARITY_STATES,
    SIMILARITY_STATE_HIDDEN,
)
from snapkv_observation_longbench import (
    add_snapkv_observation_args,
    build_snapkv_observation_run_writer,
    build_snapkv_topk_overlap_run_writer,
    validate_snapkv_observation_args,
)

model_map = load_json("config/model2path.json")
prompt_templates = load_prompt_templates()
SUPPORTED_COMPRESSION_MODEL_FAMILIES = {
    "llama",
    "mistral",
    "qwen2.5",
    "qwen3",
    "qwen3moe",
    "qwen3.5",
}


def get_model_path(model_name):
    return model_map.get(model_name, model_name)

def get_max_input_len(model_maxlen, max_new_tokens):
    max_len = model_maxlen
    if max_len is None or max_len > 10 ** 8:
        max_len = 120000
    return max(1, max_len - max_new_tokens)

def truncate_prompt(prompt, tokenizer, max_input_len):
    if max_input_len is None:
        raise ValueError("max_input_len is required.")
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > max_input_len:
        half = max_input_len // 2
        input_ids = input_ids[:half] + input_ids[-(max_input_len - half):]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt

def build_compression_config(
    compression_mode,
    compression_budget,
    hidden_mix_profile_path=None,
):
    method_config = {
        "budget": compression_budget,
        "window_size": 8,
        "mix_lambda": 0.07,
        "retain_ratio": 0.2,
        "retain_direction": "last",
        "first_tokens": 4,
    }
    if hidden_mix_profile_path:
        method_config["hidden_mix_profile_path"] = hidden_mix_profile_path

    return {
        "method": compression_mode,
        "method_config": method_config,
        "compression": None,
        "update_kv": True,
    }


def get_model_family(model_path):
    model_path_lower = model_path.lower()
    if "qwen3.5" in model_path_lower or "qwen3_5" in model_path_lower:
        return "qwen3.5"
    if "qwen3moe" in model_path_lower or "qwen3-moe" in model_path_lower or "qwen3_moe" in model_path_lower:
        return "qwen3moe"
    if "qwen3" in model_path_lower and re.search(r"[-_]a\d+b", model_path_lower):
        return "qwen3moe"
    if "qwen3" in model_path_lower:
        return "qwen3"
    if "llama" in model_path_lower:
        return "llama"
    if "mistral" in model_path_lower:
        return "mistral"
    if "qwen2.5" in model_path_lower or "qwen2_5" in model_path_lower:
        return "qwen2.5"

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "").lower().replace("-", "_")
        if model_type == "mistral":
            return "mistral"
        if model_type == "qwen2":
            return "qwen2.5"
        if model_type == "qwen3_moe":
            return "qwen3moe"
        if model_type == "qwen3":
            return "qwen3"
        if model_type == "llama":
            return "llama"
    except Exception:
        pass

    return None


def apply_compression_setup(model, tokenizer, compression_mode):
    model.config.update(
        {
            "divide_method": "step_length",
            "divide_length": 128,
            "compression_content": "think",
            "method": compression_mode,
        }
    )
    model.newline_token_ids = [
        tokenizer.encode(text, add_special_tokens=False)[-1]
        for text in ["\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n"]
    ]
    model.after_think_token_ids = [tokenizer.encode("</think>", add_special_tokens=False)[-1]]


def apply_compression_monkeypatch(model_family, compression_config):
    from models.compression.monkeypatch import replace_llama, replace_mistral, replace_qwen2_5, replace_qwen3, replace_qwen3_5, replace_qwen3moe

    if model_family == "llama":
        replace_llama(compression_config)
    elif model_family == "mistral":
        replace_mistral(compression_config)
    elif model_family == "qwen2.5":
        replace_qwen2_5(compression_config)
    elif model_family == "qwen3":
        replace_qwen3(compression_config)
    elif model_family == "qwen3moe":
        replace_qwen3moe(compression_config)
    elif model_family == "qwen3.5":
        replace_qwen3_5(compression_config)
    else:
        raise ValueError(
            f"Compression supports {sorted(SUPPORTED_COMPRESSION_MODEL_FAMILIES)}, got: {model_family}"
        )


def load_local_model(model_family, model_path, model_kwargs):
    if model_family == "llama":
        from models.llama.configuration_llama import LlamaConfig
        from models.llama.modeling_llama import LlamaForCausalLM

        config = LlamaConfig.from_pretrained(model_path)
        return LlamaForCausalLM.from_pretrained(model_path, config=config, **model_kwargs)

    if model_family == "mistral":
        from models.mistral.configuration_mistral import MistralConfig
        from models.mistral.modeling_mistral import MistralForCausalLM

        config = MistralConfig.from_pretrained(model_path)
        return MistralForCausalLM.from_pretrained(model_path, config=config, **model_kwargs)

    if model_family == "qwen2.5":
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

        config = Qwen2Config.from_pretrained(model_path)
        return Qwen2ForCausalLM.from_pretrained(model_path, config=config, **model_kwargs)

    if model_family == "qwen3":
        from models.qwen3.configuration_qwen3 import Qwen3Config
        from models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        config = Qwen3Config.from_pretrained(model_path)
        return Qwen3ForCausalLM.from_pretrained(model_path, config=config, **model_kwargs)

    if model_family == "qwen3moe":
        from models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
        from models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

        config = Qwen3MoeConfig.from_pretrained(model_path)
        return Qwen3MoeForCausalLM.from_pretrained(model_path, config=config, **model_kwargs)

    if model_family == "qwen3.5":
        from models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

        config = Qwen3_5Config.from_pretrained(model_path)
        return Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            **model_kwargs,
        )

    return AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)


def load_model_and_tokenizer(
    model_name,
    attn_heatmap_mode=False,
    compression=False,
    compression_mode=None,
    compression_budget=4096,
    hidden_mix_profile_path=None,
):
    model_path = get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32
    if attn_heatmap_mode:
        model_kwargs["attn_implementation"] = "eager"
    else:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model_family = get_model_family(model_path)
    if compression:
        if not compression_mode:
            raise ValueError("Please provide --compression_mode when --compression is enabled.")
        if model_family not in SUPPORTED_COMPRESSION_MODEL_FAMILIES:
            raise ValueError(
                "Compression currently supports llama, mistral, qwen2.5, qwen3, qwen3moe, and qwen3.5 "
                f"models, got: {model_name}"
            )

        compression_config = build_compression_config(
            compression_mode,
            compression_budget,
            hidden_mix_profile_path,
        )
        apply_compression_monkeypatch(model_family, compression_config)
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        apply_compression_setup(model, tokenizer, compression_mode)
    else:
        model = load_local_model(
            model_family,
            model_path,
            model_kwargs,
        )
    if attn_heatmap_mode:
        text_config = getattr(model.config, "text_config", model.config)
        setattr(text_config, "_attn_implementation", "eager")
        setattr(model.config, "_attn_implementation", "eager")
    model.eval()
    return model, tokenizer

def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_inputs(prompt, tokenizer, device, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            inputs = input_ids if isinstance(input_ids, dict) else {"input_ids": input_ids}
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    inputs = dict(inputs)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return {k: v.to(device) for k, v in inputs.items()}

def query_llm(
    prompt,
    model_name,
    model,
    tokenizer,
    model_maxlen,
    temperature=0.5,
    max_new_tokens=128,
    stop=None,
    enable_thinking=False,
    attn_sample_writer=None,
    attn_output_ratio_sample_writer=None,
    query_window_sample_writer=None,
    snapkv_observation_sample_writer=None,
    snapkv_topk_overlap_sample_writer=None,
    hidden_state_pca_sample_writer=None,
    prefill_label="response",
):
    max_input_len = get_max_input_len(model_maxlen, max_new_tokens)
    prompt = truncate_prompt(prompt, tokenizer, max_input_len)
    device = get_input_device(model)
    inputs = build_inputs(prompt, tokenizer, device, enable_thinking=enable_thinking)
    input_len = inputs["input_ids"].shape[-1]
    capture_record = None

    if attn_sample_writer is not None:
        capture_record = attn_sample_writer.capture_prefill(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            inputs=inputs,
            label=prefill_label,
        ).record
    if query_window_sample_writer is not None:
        query_window_sample_writer.capture_prefill(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            inputs=inputs,
            label=prefill_label,
        )
    if attn_output_ratio_sample_writer is not None:
        attn_output_ratio_sample_writer.capture_prefill(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            inputs=inputs,
            label=prefill_label,
        )
    if snapkv_observation_sample_writer is not None:
        snapkv_observation_sample_writer.capture_prefill(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            inputs=inputs,
            label=prefill_label,
        )
    if snapkv_topk_overlap_sample_writer is not None:
        snapkv_topk_overlap_sample_writer.capture_prefill(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            inputs=inputs,
            label=prefill_label,
        )
    if hidden_state_pca_sample_writer is not None:
        hidden_state_pca_sample_writer.capture_prefill(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            inputs=inputs,
            label=prefill_label,
        )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)[0]
    return tokenizer.decode(output_ids[input_len:], skip_special_tokens=True), capture_record

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def build_prompt(template, context, item, cot=None):
    prompt = (
        template.replace('$DOC$', context.strip())
        .replace('$Q$', item['question'].strip())
        .replace('$C_A$', item['choice_A'].strip())
        .replace('$C_B$', item['choice_B'].strip())
        .replace('$C_C$', item['choice_C'].strip())
        .replace('$C_D$', item['choice_D'].strip())
    )
    if cot is not None:
        prompt = prompt.replace('$COT$', cot)
    return prompt

def build_attn_run_writer(args, out_file, model):
    if not args.attn_heatmap_mode:
        return None
    full_attention_layers = get_full_attention_layer_indices(model)
    return AttentionHeatmapRunWriter(
        root_dir=args.attn_heatmap_dir,
        model_name=args.model,
        out_file=out_file,
        full_attention_layers=full_attention_layers,
        max_prefill_tokens=args.attn_max_prefill_tokens,
    )


def build_query_window_similarity_run_writer(args, out_file):
    if not args.query_window_similarity_mode:
        return None
    return QueryWindowSimilarityRunWriter(
        root_dir=args.query_window_similarity_dir,
        model_name=args.model,
        out_file=out_file,
        window_size=args.query_window_size,
        max_prefill_tokens=args.query_window_max_prefill_tokens,
        similarity_state=args.query_window_similarity_submode,
    )


def build_attn_output_ratio_run_writer(args, out_file):
    if not args.attn_output_ratio_mode:
        return None
    return AttnOutputRatioRunWriter(
        root_dir=args.attn_output_ratio_dir,
        model_name=args.model,
        out_file=out_file,
        max_prefill_tokens=args.attn_output_ratio_max_prefill_tokens,
        layer_spec=args.attn_output_ratio_layers,
    )


def build_hidden_state_pca_run_writer(args, out_file):
    if not args.hidden_state_pca_mode:
        return None
    return HiddenStatePCARunWriter(
        root_dir=args.hidden_state_pca_dir,
        model_name=args.model,
        out_file=out_file,
        max_prefill_tokens=args.hidden_state_pca_max_prefill_tokens,
        layer_spec=args.hidden_state_pca_layers,
        token_start=args.hidden_state_pca_token_start,
        token_end=args.hidden_state_pca_token_end,
        submode=args.hidden_state_pca_submode,
    )


def validate_args(args):
    if args.model_maxlen < 1:
        raise ValueError("--model_maxlen must be at least 1.")
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("--num_samples must be at least 1 when provided.")
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
    if args.hidden_state_pca_submode not in SUPPORTED_HIDDEN_STATE_PCA_SUBMODES:
        raise ValueError(
            "--hidden_state_pca_submode must be one of "
            f"{sorted(SUPPORTED_HIDDEN_STATE_PCA_SUBMODES)}."
        )
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

def get_pred(data, args, fout, out_file):
    model_name = args.model
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        attn_heatmap_mode=args.attn_heatmap_mode,
        compression=args.compression,
        compression_mode=args.compression_mode,
        compression_budget=args.compression_budget,
        hidden_mix_profile_path=args.hidden_mix_profile_path,
    )
    attn_run_writer = build_attn_run_writer(args, out_file, model)
    query_window_run_writer = build_query_window_similarity_run_writer(args, out_file)
    attn_output_ratio_run_writer = build_attn_output_ratio_run_writer(args, out_file)
    snapkv_observation_run_writer = build_snapkv_observation_run_writer(args, out_file)
    snapkv_topk_overlap_run_writer = build_snapkv_topk_overlap_run_writer(args, out_file)
    hidden_state_pca_run_writer = build_hidden_state_pca_run_writer(args, out_file)
    for sample_index, item in enumerate(tqdm(data)):
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
        snapkv_topk_overlap_sample_writer = (
            snapkv_topk_overlap_run_writer.new_sample(item)
            if snapkv_topk_overlap_run_writer is not None
            else None
        )
        hidden_state_pca_sample_writer = (
            hidden_state_pca_run_writer.new_sample(item)
            if hidden_state_pca_run_writer is not None
            else None
        )
        try:
            context = item['context']
            if args.rag > 0:
                template = prompt_templates["rag"]
                retrieved = item["retrieved_context"][:args.rag]
                retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
                context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
            elif args.no_context:
                template = prompt_templates["no_context"]
            elif args.cot:
                template = prompt_templates["cot"]
            else:
                template = prompt_templates["zero_shot"]
            prompt = build_prompt(template, context, item)
            if args.cot:
                output, _ = query_llm(
                    prompt,
                    model_name,
                    model,
                    tokenizer,
                    args.model_maxlen,
                    temperature=0,
                    max_new_tokens=1024,
                    enable_thinking=args.cot,
                    attn_sample_writer=attn_sample_writer,
                    attn_output_ratio_sample_writer=attn_output_ratio_sample_writer,
                    query_window_sample_writer=query_window_sample_writer,
                    snapkv_observation_sample_writer=snapkv_observation_sample_writer,
                    snapkv_topk_overlap_sample_writer=snapkv_topk_overlap_sample_writer,
                    hidden_state_pca_sample_writer=hidden_state_pca_sample_writer,
                    prefill_label="cot_reasoning",
                )
            else:
                output, _ = query_llm(
                    prompt,
                    model_name,
                    model,
                    tokenizer,
                    args.model_maxlen,
                    temperature=0,
                    max_new_tokens=128,
                    enable_thinking=args.cot,
                    attn_sample_writer=attn_sample_writer,
                    attn_output_ratio_sample_writer=attn_output_ratio_sample_writer,
                    query_window_sample_writer=query_window_sample_writer,
                    snapkv_observation_sample_writer=snapkv_observation_sample_writer,
                    snapkv_topk_overlap_sample_writer=snapkv_topk_overlap_sample_writer,
                    hidden_state_pca_sample_writer=hidden_state_pca_sample_writer,
                    prefill_label="response",
                )
            if output == '':
                continue
            if args.cot: # extract answer
                response = output.strip()
                item['response_cot'] = response
                prompt = build_prompt(prompt_templates["cot_answer"], context, item, cot=response)
                output, _ = query_llm(
                    prompt,
                    model_name,
                    model,
                    tokenizer,
                    args.model_maxlen,
                    temperature=0.1,
                    max_new_tokens=128,
                    enable_thinking=args.cot,
                    attn_sample_writer=attn_sample_writer,
                    attn_output_ratio_sample_writer=attn_output_ratio_sample_writer,
                    query_window_sample_writer=query_window_sample_writer,
                    snapkv_observation_sample_writer=snapkv_observation_sample_writer,
                    snapkv_topk_overlap_sample_writer=snapkv_topk_overlap_sample_writer,
                    hidden_state_pca_sample_writer=hidden_state_pca_sample_writer,
                    prefill_label="cot_answer_extraction",
                )
                if output == '':
                    continue
            response = output.strip()
            item['response'] = response
            item['pred'] = extract_answer(response)
            item['judge'] = item['pred'] == item['answer']
            item['context'] = context[:1000]
            if attn_sample_writer is not None:
                item["attn_capture_status"] = attn_sample_writer.build_capture_status()
                item["attn_artifact"] = os.path.relpath(attn_sample_writer.sample_dir, start=args.attn_heatmap_dir)
            if query_window_sample_writer is not None:
                item["query_window_similarity_status"] = query_window_sample_writer.build_capture_status()
                item["query_window_similarity_artifact"] = os.path.relpath(
                    query_window_sample_writer.sample_dir,
                    start=args.query_window_similarity_dir,
                )
            if attn_output_ratio_sample_writer is not None:
                item["attn_output_ratio_status"] = attn_output_ratio_sample_writer.build_capture_status()
                item["attn_output_ratio_artifact"] = os.path.relpath(
                    attn_output_ratio_sample_writer.sample_dir,
                    start=args.attn_output_ratio_dir,
                )
            if snapkv_observation_sample_writer is not None:
                item["snapkv_observation_status"] = snapkv_observation_sample_writer.build_capture_status()
                item["snapkv_observation_artifact"] = os.path.relpath(
                    snapkv_observation_sample_writer.sample_dir,
                    start=args.snapkv_observation_dir,
                )
            if snapkv_topk_overlap_sample_writer is not None:
                item["snapkv_topk_overlap_status"] = snapkv_topk_overlap_sample_writer.build_capture_status()
                item["snapkv_topk_overlap_artifact"] = os.path.relpath(
                    snapkv_topk_overlap_sample_writer.sample_dir,
                    start=args.snapkv_observation_dir,
                )
            if hidden_state_pca_sample_writer is not None:
                item["hidden_state_pca_status"] = hidden_state_pca_sample_writer.build_capture_status()
                item["hidden_state_pca_artifact"] = os.path.relpath(
                    hidden_state_pca_sample_writer.sample_dir,
                    start=args.hidden_state_pca_dir,
                )
            if attn_sample_writer is not None:
                attn_sample_writer.finalize(item)
            if query_window_sample_writer is not None:
                query_window_sample_writer.finalize(item)
            if attn_output_ratio_sample_writer is not None:
                attn_output_ratio_sample_writer.finalize(item)
            if snapkv_observation_sample_writer is not None:
                snapkv_observation_sample_writer.finalize(item)
            if snapkv_topk_overlap_sample_writer is not None:
                snapkv_topk_overlap_sample_writer.finalize(item)
            if hidden_state_pca_sample_writer is not None:
                hidden_state_pca_sample_writer.finalize(item)
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            fout.flush()
        except Exception as exc:
            sample_id = item.get("_id", "unknown")
            print(f"Skipping sample {sample_id} due to error: {exc}")
            traceback.print_exc()
            if attn_sample_writer is not None:
                item["error"] = str(exc)
                item["attn_capture_status"] = attn_sample_writer.build_capture_status()
                item["attn_artifact"] = os.path.relpath(attn_sample_writer.sample_dir, start=args.attn_heatmap_dir)
                attn_sample_writer.finalize(item)
            if query_window_sample_writer is not None:
                item["error"] = str(exc)
                item["query_window_similarity_status"] = query_window_sample_writer.build_capture_status()
                item["query_window_similarity_artifact"] = os.path.relpath(
                    query_window_sample_writer.sample_dir,
                    start=args.query_window_similarity_dir,
                )
                query_window_sample_writer.finalize(item)
            if attn_output_ratio_sample_writer is not None:
                item["error"] = str(exc)
                item["attn_output_ratio_status"] = attn_output_ratio_sample_writer.build_capture_status()
                item["attn_output_ratio_artifact"] = os.path.relpath(
                    attn_output_ratio_sample_writer.sample_dir,
                    start=args.attn_output_ratio_dir,
                )
                attn_output_ratio_sample_writer.finalize(item)
            if snapkv_observation_sample_writer is not None:
                item["error"] = str(exc)
                item["snapkv_observation_status"] = snapkv_observation_sample_writer.build_capture_status()
                item["snapkv_observation_artifact"] = os.path.relpath(
                    snapkv_observation_sample_writer.sample_dir,
                    start=args.snapkv_observation_dir,
                )
                snapkv_observation_sample_writer.finalize(item)
            if snapkv_topk_overlap_sample_writer is not None:
                item["error"] = str(exc)
                item["snapkv_topk_overlap_status"] = snapkv_topk_overlap_sample_writer.build_capture_status()
                item["snapkv_topk_overlap_artifact"] = os.path.relpath(
                    snapkv_topk_overlap_sample_writer.sample_dir,
                    start=args.snapkv_observation_dir,
                )
                snapkv_topk_overlap_sample_writer.finalize(item)
            if hidden_state_pca_sample_writer is not None:
                item["error"] = str(exc)
                item["hidden_state_pca_status"] = hidden_state_pca_sample_writer.build_capture_status()
                item["hidden_state_pca_artifact"] = os.path.relpath(
                    hidden_state_pca_sample_writer.sample_dir,
                    start=args.hidden_state_pca_dir,
                )
                hidden_state_pca_sample_writer.finalize(item)
            continue


def main(args):
    print(args)
    validate_args(args)
    domains = parse_domain_filter(args.domain)
    out_file = build_output_path(args, domains)
    print(f"Writing results to {out_file}")

    data_all = load_longbench_v2(domains)
    processed_ids = load_processed_ids(out_file)
    data = select_unprocessed(data_all, processed_ids)
    if args.num_samples is not None:
        data = data[:args.num_samples]
        print(f"Limited this run to {len(data)} example(s).")

    if len(data) == 0:
        print("No new examples to process.")
        return

    with open(out_file, 'a', encoding='utf-8') as fout:
        if args.n_proc == 1:
            get_pred(data, args, fout, out_file)
        else:
            print("Warning: each process will load its own transformers model copy.")
            data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
            processes = []
            for rank in range(args.n_proc):
                p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout, out_file))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="output_dir/results_longbench")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--model_maxlen", type=int, default=120000, help="Model context length used for prompt truncation.")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--domain", "-d", action='append', default=None, help="Only run examples from the given domain. Repeat this option or use comma-separated names for multiple domains.")
    parser.add_argument("--num_samples", "--max_samples", type=int, default=None, help="Only run the first N unprocessed examples.")
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--compression_mode", type=str, default=None)
    parser.add_argument("--compression_budget", type=int, default=4096)
    parser.add_argument("--hidden_mix_profile_path", type=str, default=None)
    parser.add_argument("--attn_heatmap_mode", action="store_true")
    parser.add_argument("--attn_heatmap_dir", type=str, default="output_dir/results_longbench/attn_heatmaps")
    parser.add_argument("--attn_max_prefill_tokens", type=int, default=None, help="Skip attention heatmap capture when the prefill token count exceeds this cap.")
    parser.add_argument("--query_window_similarity_mode", action="store_true")
    parser.add_argument("--query_window_similarity_dir", type=str, default="output_dir/results_longbench/query_window_similarity")
    parser.add_argument("--query_window_size", type=int, default=8, help="Number of prompt-tail tokens used for layer-wise query-window analysis.")
    parser.add_argument(
        "--query_window_similarity_submode",
        "--query_window_similarity_state",
        dest="query_window_similarity_submode",
        type=str,
        choices=sorted(SUPPORTED_SIMILARITY_STATES),
        default=SIMILARITY_STATE_HIDDEN,
        help="Submode for query-window layer analysis: hidden_states/query_states use cosine similarity; hidden_states_l2_diff computes pairwise L2 norms of hidden-state differences.",
    )
    parser.add_argument("--query_window_max_prefill_tokens", type=int, default=None, help="Skip query window similarity capture when the prefill token count exceeds this cap.")
    parser.add_argument("--attn_output_ratio_mode", action="store_true", help="Capture per-layer per-token ||attn_output||_2 / ||hidden_states||_2 ratios during prefill and plot density curves.")
    parser.add_argument("--attn_output_ratio_dir", type=str, default="output_dir/results_longbench/attn_output_ratios")
    parser.add_argument("--attn_output_ratio_layers", type=str, default="all", help="Layers to visualize: all, auto, comma-separated ids, or ranges like 5,10,20-25. Raw npz always stores every captured layer.")
    parser.add_argument("--attn_output_ratio_max_prefill_tokens", type=int, default=None, help="Skip attention-output ratio capture when the prefill token count exceeds this cap.")
    parser.add_argument("--hidden_state_pca_mode", action="store_true", help="Capture a token span during prefill and plot shared-PCA 2D scatters by layer.")
    parser.add_argument(
        "--hidden_state_pca_submode",
        type=str,
        choices=sorted(SUPPORTED_HIDDEN_STATE_PCA_SUBMODES),
        default=PCA_SUBMODE_KEY_VALUE_STATES,
        help="Submode for hidden-state PCA capture: key_value_states keeps existing layer-wise key/value plots; key_value_heads plots per-layer key/value head PCA distributions; hidden_states captures decoder layer outputs.",
    )
    parser.add_argument("--hidden_state_pca_dir", type=str, default="output_dir/results_longbench/hidden_state_pca")
    parser.add_argument("--hidden_state_pca_layers", type=str, default="all", help="Layers to visualize: all, auto, comma-separated ids, or ranges like 5,10,20-25.")
    parser.add_argument("--hidden_state_pca_token_start", type=int, default=0, help="Start token index for hidden-state PCA span. Negative values count from the prompt end.")
    parser.add_argument("--hidden_state_pca_token_end", type=int, default=None, help="Exclusive end token index for hidden-state PCA span. Defaults to the prompt end; negative values count from the prompt end.")
    parser.add_argument("--hidden_state_pca_max_prefill_tokens", type=int, default=None, help="Skip hidden-state PCA capture when the prefill token count exceeds this cap.")
    add_snapkv_observation_args(parser)
    args = parser.parse_args()
    main(args)
