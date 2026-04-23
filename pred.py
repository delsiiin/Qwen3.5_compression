import argparse
import json
import re

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pred_misc import (
    build_output_path,
    load_json,
    load_longbench_v2,
    load_processed_ids,
    load_prompt_templates,
    parse_domain_filter,
    select_unprocessed,
)

model_map = load_json("config/model2path.json")
maxlen_map = load_json("config/model2maxlen.json")
prompt_templates = load_prompt_templates()


def get_model_path(model_name):
    return model_map.get(model_name, model_name)

def get_max_input_len(model_name, tokenizer, max_new_tokens):
    max_len = maxlen_map.get(model_name, getattr(tokenizer, "model_max_length", 120000))
    if max_len is None or max_len > 10 ** 8:
        max_len = 120000
    return max(1, max_len - max_new_tokens)

def truncate_prompt(prompt, model_name, tokenizer, max_new_tokens):
    max_input_len = get_max_input_len(model_name, tokenizer, max_new_tokens)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > max_input_len:
        half = max_input_len // 2
        input_ids = input_ids[:half] + input_ids[-(max_input_len - half):]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt

def load_model_and_tokenizer(model_name):
    model_path = get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if "qwen3.5" in model_name.lower():
        from models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", **model_kwargs)
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

def query_llm(prompt, model_name, model, tokenizer, temperature=0.5, max_new_tokens=128, stop=None, enable_thinking=False):
    prompt = truncate_prompt(prompt, model_name, tokenizer, max_new_tokens)
    device = get_input_device(model)
    inputs = build_inputs(prompt, tokenizer, device, enable_thinking=enable_thinking)
    input_len = inputs["input_ids"].shape[-1]

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
    return tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)

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

def get_pred(data, args, fout):
    model_name = args.model
    model, tokenizer = load_model_and_tokenizer(model_name)
    for item in tqdm(data):
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
            output = query_llm(prompt, model_name, model, tokenizer, temperature=0.1, max_new_tokens=1024, enable_thinking=args.cot)
        else:
            output = query_llm(prompt, model_name, model, tokenizer, temperature=0.1, max_new_tokens=128, enable_thinking=args.cot)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = build_prompt(prompt_templates["cot_answer"], context, item, cot=response)
            output = query_llm(prompt, model_name, model, tokenizer, temperature=0.1, max_new_tokens=128, enable_thinking=args.cot)
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main(args):
    print(args)
    domains = parse_domain_filter(args.domain)
    out_file = build_output_path(args, domains)
    print(f"Writing results to {out_file}")

    data_all = load_longbench_v2(domains)
    processed_ids = load_processed_ids(out_file)
    data = select_unprocessed(data_all, processed_ids)

    if len(data) == 0:
        print("No new examples to process.")
        return

    with open(out_file, 'a', encoding='utf-8') as fout:
        if args.n_proc == 1:
            get_pred(data, args, fout)
        else:
            print("Warning: each process will load its own transformers model copy.")
            data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
            processes = []
            for rank in range(args.n_proc):
                p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--domain", "-d", action='append', default=None, help="Only run examples from the given domain. Repeat this option or use comma-separated names for multiple domains.")
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    args = parser.parse_args()
    main(args)
