import os, csv, json
import argparse
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
import re
import importlib.util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.multiprocessing as mp

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def parse_domain_filter(domain_args):
    if not domain_args:
        return None
    domains = []
    for domain_arg in domain_args:
        for domain in domain_arg.split(','):
            domain = domain.strip()
            if domain and domain not in domains:
                domains.append(domain)
    return domains or None

def normalize_domain(domain):
    return domain.strip().lower()

def get_domain_suffix(domains):
    if not domains:
        return ''
    safe_domains = []
    for domain in domains:
        safe_domain = re.sub(r'[^A-Za-z0-9._-]+', '_', domain).strip('_')
        if safe_domain:
            safe_domains.append(safe_domain)
    return '_domain_' + '_'.join(safe_domains) if safe_domains else ''

def filter_by_domain(data, domains):
    if not domains:
        return data
    domain_set = {normalize_domain(domain) for domain in domains}
    available_domains = sorted({item['domain'].strip() for item in data})
    available_domain_set = {normalize_domain(domain) for domain in available_domains}
    missing_domains = [domain for domain in domains if normalize_domain(domain) not in available_domain_set]
    if missing_domains:
        raise ValueError(
            f"Domain(s) not found: {', '.join(missing_domains)}. "
            f"Available domains: {', '.join(available_domains)}"
        )
    filtered_data = [item for item in data if normalize_domain(item['domain']) in domain_set]
    print(f"Selected {len(filtered_data)} / {len(data)} examples for domain(s): {', '.join(domains)}")
    return filtered_data

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

def get_pred(data, args, fout):
    model_name = args.model
    model, tokenizer = load_model_and_tokenizer(model_name)
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        if args.cot:
            output = query_llm(prompt, model_name, model, tokenizer, temperature=0.1, max_new_tokens=1024, enable_thinking=args.cot)
        else:
            output = query_llm(prompt, model_name, model, tokenizer, temperature=0.1, max_new_tokens=128, enable_thinking=args.cot)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
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

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    domains = parse_domain_filter(args.domain)
    output_prefix = args.model.split("/")[-1] + get_domain_suffix(domains)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, output_prefix + f"_rag_{str(args.rag)}_{timestamp}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, output_prefix + f"_no_context_{timestamp}.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, output_prefix + f"_cot_{timestamp}.jsonl")
    else:
        out_file = os.path.join(args.save_dir, output_prefix + f"_{timestamp}.jsonl")
    print(f"Writing results to {out_file}")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]
    data_all = filter_by_domain(data_all, domains)

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)
    if len(data) == 0:
        print("No new examples to process.")
        fout.close()
        return

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
    main()
